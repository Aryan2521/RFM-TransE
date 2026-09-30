import argparse
import math
import random
import time
import torch
import torch.optim as optim
import numpy as np

from transe import TransE
from RFM_TransE import RFMTransE
from utils import (
    RealKG,
    generate_training_negatives,
    generate_fuzzy_labels,
    get_structural_fuzzy_base,
    evaluate_link_prediction,
    optimize_thresholds,
    evaluate_triple_classification,
    compute_fuzzy_evaluation_metrics
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_fuzzy_scores(model, test_triples, test_neg, device, batch_size=4096):
    """
    Computes fuzzy scores using the trained model.
    """
    model.eval()

    def predict(triples):
        predictions = []
        with torch.no_grad():
            for start in range(0, len(triples), batch_size):
                batch = torch.tensor(
                    triples[start:start + batch_size], dtype=torch.long, device=device
                )
                predictions.append(model.fuzzy_score(*batch.unbind(dim=1)).cpu().numpy())
        return np.concatenate(predictions) if predictions else np.empty(0)

    return predict(test_triples), predict(test_neg)


def split_fraction(value):
    value = float(value)
    if not 0 < value <= 1:
        raise argparse.ArgumentTypeError("Split fractions must be greater than 0 and at most 1.")
    return value


def positive_integer(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Batch size must be positive.")
    return value


def select_split(triples, fraction=1.0, cap=0, seed=42):
    """Select within an official split, without mutating it or global RNG state.

    Fractions apply before optional positive caps. Nonpositive caps mean no cap.
    Returning a copy, even for full coverage, protects the cached dataset order.
    """
    if not 0 < fraction <= 1:
        raise ValueError("Split fractions must be greater than 0 and at most 1.")
    if not triples:
        raise ValueError("Cannot benchmark an empty official split.")
    count = max(1, math.ceil(len(triples) * fraction))
    if cap > 0:
        count = min(count, cap)
    if count == len(triples):
        return list(triples)
    indices = sorted(random.Random(seed).sample(range(len(triples)), count))
    return [triples[index] for index in indices]


def prepare_splits(kg, args, seed):
    """Keep official train/validation/test boundaries; optional sampling is local."""
    val_cap = args.eval_sample_size if args.val_sample_size is None else args.val_sample_size
    test_cap = args.eval_sample_size if args.test_sample_size is None else args.test_sample_size
    return (
        select_split(kg.train_triples, args.train_fraction, args.train_sample_size, seed),
        select_split(kg.val_triples, args.val_fraction, val_cap, seed + 1),
        select_split(kg.test_triples, args.test_fraction, test_cap, seed + 2),
    )


def compute_training_coverage(full_train_triples, sampled_train_triples):
    """Measure how much of the original training graph remains after sampling."""
    full_entities = {entity for h, _, t in full_train_triples for entity in (h, t)}
    sampled_entities = {entity for h, _, t in sampled_train_triples for entity in (h, t)}
    full_relations = {r for _, r, _ in full_train_triples}
    sampled_relations = {r for _, r, _ in sampled_train_triples}

    def percentage(observed, total):
        return 100.0 * observed / total if total else 0.0

    return {
        "Full Train Triples": len(full_train_triples),
        "Sampled Train Triples": len(sampled_train_triples),
        "Triple Coverage (%)": percentage(
            len(sampled_train_triples), len(full_train_triples)
        ),
        "Full Train Entities": len(full_entities),
        "Sampled Train Entities": len(sampled_entities),
        "Entity Coverage (%)": percentage(
            len(sampled_entities), len(full_entities)
        ),
        "Full Train Relations": len(full_relations),
        "Sampled Train Relations": len(sampled_relations),
        "Relation Coverage (%)": percentage(
            len(sampled_relations), len(full_relations)
        ),
    }


def summarize_target_distribution(values, base_value):
    """Return descriptive statistics for one fixed group of synthetic targets."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {
            "Count": 0,
            "Mean": float("nan"),
            "Within-Set SD": float("nan"),
            "Min": float("nan"),
            "Median": float("nan"),
            "Max": float("nan"),
            "Structurally Adjusted (%)": float("nan"),
        }

    return {
        "Count": int(values.size),
        "Mean": float(np.mean(values)),
        "Within-Set SD": float(np.std(values, ddof=0)),
        "Min": float(np.min(values)),
        "Median": float(np.median(values)),
        "Max": float(np.max(values)),
        "Structurally Adjusted (%)": float(
            100.0 * np.mean(~np.isclose(values, base_value, rtol=0.0, atol=1e-6))
        ),
    }


def aggregate_seed_statistic(values):
    """Aggregate a statistic using mean and sample SD across random seeds."""
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
    }

def run_single_seed_experiment(dataset_name, uncertainty_level, seed, args, device, kg, all_triples_set, num_entities, num_relations):
    set_seed(seed)
    
    # Use complete official splits unless the user explicitly requests subsets.
    train_triples, val_eval_subset, test_eval_subset = prepare_splits(kg, args, seed)
    for name, full, selected in (
        ("Train", kg.train_triples, train_triples),
        ("Validation", kg.val_triples, val_eval_subset),
        ("Test", kg.test_triples, test_eval_subset),
    ):
        print(f"{name}: {len(selected):,}/{len(full):,} triples ({100 * len(selected) / len(full):.2f}%)")
        
    # Adjacency list for Jaccard-based fuzzy label generation
    entity_neighbors = {i: set() for i in range(num_entities)}
    for h, r, t in train_triples:
        entity_neighbors[h].add(t)
        entity_neighbors[t].add(h)
        
    # Generate validation and test negatives
    val_neg = kg.get_corrupted_triples(val_eval_subset, all_triples_set, num_entities)
    test_neg = kg.get_corrupted_triples(test_eval_subset, all_triples_set, num_entities)

    # Continuous targets are fixed within a seed and shared by every model.
    val_target_pos = generate_fuzzy_labels(
        val_eval_subset, True, uncertainty_level, "cpu", entity_neighbors
    ).numpy()
    val_target_neg = generate_fuzzy_labels(
        val_neg, False, uncertainty_level, "cpu", entity_neighbors
    ).numpy()

    test_target_pos = generate_fuzzy_labels(
        test_eval_subset, True, uncertainty_level, "cpu", entity_neighbors
    ).numpy()
    test_target_neg = generate_fuzzy_labels(
        test_neg, False, uncertainty_level, "cpu", entity_neighbors
    ).numpy()
    test_target_all = np.concatenate([test_target_pos, test_target_neg])

    # Audit the sampled graph and every fixed target set used in evaluation.
    train_target_pos = generate_fuzzy_labels(
        train_triples, True, uncertainty_level, "cpu", entity_neighbors
    ).numpy()
    positive_base = get_structural_fuzzy_base(uncertainty_level, True)
    negative_base = get_structural_fuzzy_base(uncertainty_level, False)
    experiment_statistics = {
        "Training Coverage": compute_training_coverage(
            kg.train_triples, train_triples
        ),
        "Target Distributions": {
            "Training Positive": summarize_target_distribution(
                train_target_pos, positive_base
            ),
            "Validation Positive": summarize_target_distribution(
                val_target_pos, positive_base
            ),
            "Validation Negative": summarize_target_distribution(
                val_target_neg, negative_base
            ),
            "Test Positive": summarize_target_distribution(
                test_target_pos, positive_base
            ),
            "Test Negative": summarize_target_distribution(
                test_target_neg, negative_base
            ),
        },
    }
    # Include validation/test counts and vocabulary coverage in the same audit.
    for name, full, selected in (
        ("Validation", kg.val_triples, val_eval_subset),
        ("Test", kg.test_triples, test_eval_subset),
    ):
        for key, value in compute_training_coverage(full, selected).items():
            experiment_statistics["Training Coverage"][f"{name} {key}"] = value
    
    # Instantiate only the requested models. This keeps focused reruns efficient.
    model_factories = {
        "standard": lambda: TransE(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=args.embedding_dim,
            margin=args.margin,
            p_norm=args.p_norm
        ).to(device),

        "sigmoid": lambda: RFMTransE(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=args.embedding_dim,
            margin=args.margin,
            p_norm=args.p_norm,
            alpha=args.alpha,
            lambda_fuzzy=args.lambda_fuzzy,
            fuzzy_bias=args.fuzzy_bias,
            membership_function="sigmoid"
        ).to(device),

        "exponential": lambda: RFMTransE(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=args.embedding_dim,
            margin=args.margin,
            p_norm=args.p_norm,
            alpha=args.alpha,
            lambda_fuzzy=args.lambda_fuzzy,
            fuzzy_bias=args.fuzzy_bias,
            membership_function="exponential"
        ).to(device),

        "gaussian": lambda: RFMTransE(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=args.embedding_dim,
            margin=args.margin,
            p_norm=args.p_norm,
            alpha=args.alpha,
            lambda_fuzzy=args.lambda_fuzzy,
            fuzzy_bias=args.fuzzy_bias,
            membership_function="gaussian"
        ).to(device)
    }
    display_names = {
        "standard": "Standard TransE",
        "sigmoid": "RFM-TransE (Sigmoid)",
        "exponential": "RFM-TransE (Exponential)",
        "gaussian": "RFM-TransE (Gaussian)",
    }
    requested_models = getattr(
        args, "models", ["standard", "sigmoid", "exponential", "gaussian"]
    )
    models = {
        display_names[name]: model_factories[name]() for name in requested_models
    }
    
    seed_results = {}
    
    for model_name, model in models.items():
        set_seed(seed)
        # Each model starts from the same order; never shuffle kg.train_triples.
        model_train_triples = list(train_triples)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        model.train()
        start_time = time.time()
        
        for epoch in range(1, args.epochs + 1):
            random.shuffle(model_train_triples)
            
            for i in range(0, len(model_train_triples), args.batch_size):
                batch_pos = model_train_triples[i : i + args.batch_size]
                if len(batch_pos) == 0:
                    continue
                    
                batch_neg = generate_training_negatives(batch_pos, num_entities)
                
                pos_h = torch.tensor([t[0] for t in batch_pos], dtype=torch.long, device=device)
                pos_r = torch.tensor([t[1] for t in batch_pos], dtype=torch.long, device=device)
                pos_t = torch.tensor([t[2] for t in batch_pos], dtype=torch.long, device=device)
                
                neg_h = torch.tensor([t[0] for t in batch_neg], dtype=torch.long, device=device)
                neg_r = torch.tensor([t[1] for t in batch_neg], dtype=torch.long, device=device)
                neg_t = torch.tensor([t[2] for t in batch_neg], dtype=torch.long, device=device)
                
                pos_labels = generate_fuzzy_labels(batch_pos, True, uncertainty_level, device, entity_neighbors)
                neg_labels = generate_fuzzy_labels(batch_neg, False, uncertainty_level, device, entity_neighbors)
                
                model.train_step(
                    pos_h, pos_r, pos_t, neg_h, neg_r, neg_t, 
                    pos_labels, neg_labels, optimizer
                )
            if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
                print(f"{model_name}: epoch {epoch}/{args.epochs} complete", flush=True)
                
        training_time = time.time() - start_time
        
        model.eval()
        lp = evaluate_link_prediction(
            model, test_eval_subset, all_triples_set, num_entities, device,
            batch_size=args.eval_batch_size
        )
        val_thresholds = optimize_thresholds(
            model, val_eval_subset, val_neg, num_relations, device
        )
        tc = evaluate_triple_classification(
            model, test_eval_subset, test_neg, val_thresholds, device
        )
        pos_preds, neg_preds = evaluate_fuzzy_scores(model, test_eval_subset, test_neg, device)
        pred_all = np.concatenate([pos_preds, neg_preds])

        fuzzy_metrics = compute_fuzzy_evaluation_metrics(pred_all, test_target_all)

        result_name = model_name
        if model_name == "Standard TransE":
            result_name = "Standard TransE (Fixed Mapping Reference)"

        seed_results[result_name] = {
            "MRR": lp["Mean Reciprocal Rank"],
            "Hits@3": lp["Hits@3"],
            "Hits@5": lp["Hits@5"],
            "Precision": tc["Precision"],
            "Recall": tc["Recall"],
            "F1": tc["F1 Score"],
            "MSE": fuzzy_metrics["MSE"],
            "MAE": fuzzy_metrics["MAE"],
            "ECE": fuzzy_metrics["ECE"],
            "Train Time (s)": training_time,
        }
        
    return seed_results, experiment_statistics

def run_experiment(dataset_name, uncertainty_level, args, device):
    print(f"\n" + "="*80)
    print(f"EXPERIMENT GROUP: {dataset_name} | UNCERTAINTY: {uncertainty_level.upper()} | SEEDS: {args.seeds}")
    print("="*80)
    
    kg = RealKG(dataset_name)
    num_entities = len(kg.entity_to_id)
    num_relations = len(kg.relation_to_id)
    all_triples_set = set(kg.train_triples + kg.val_triples + kg.test_triples)
    
    print(f"Dataset stats: {num_entities} entities, {num_relations} relations.")
    print(f"Splits size: train={len(kg.train_triples)}, val={len(kg.val_triples)}, test={len(kg.test_triples)}")
    
    all_seed_results = []
    all_seed_statistics = []
    for seed in args.seeds:
        print(f"\n--- Running Seed {seed} ---")
        res_seed, stats_seed = run_single_seed_experiment(
            dataset_name, uncertainty_level, seed, args, device, kg, all_triples_set, num_entities, num_relations
        )
        all_seed_results.append(res_seed)
        all_seed_statistics.append(stats_seed)

    coverage_keys = list(all_seed_statistics[0]["Training Coverage"].keys())
    aggregated_coverage = {
        "Dataset": dataset_name,
        "Uncertainty": uncertainty_level,
    }
    for key in coverage_keys:
        aggregated = aggregate_seed_statistic([
            stats["Training Coverage"][key] for stats in all_seed_statistics
        ])
        aggregated_coverage[f"{key}_mean"] = aggregated["mean"]
        aggregated_coverage[f"{key}_std"] = aggregated["std"]

    aggregated_target_distributions = []
    target_sets = list(all_seed_statistics[0]["Target Distributions"].keys())
    target_statistic_keys = list(
        all_seed_statistics[0]["Target Distributions"][target_sets[0]].keys()
    )
    for target_set in target_sets:
        row = {
            "Dataset": dataset_name,
            "Uncertainty": uncertainty_level,
            "Target Set": target_set,
        }
        for key in target_statistic_keys:
            aggregated = aggregate_seed_statistic([
                stats["Target Distributions"][target_set][key]
                for stats in all_seed_statistics
            ])
            row[f"{key}_mean"] = aggregated["mean"]
            row[f"{key}_std"] = aggregated["std"]
        aggregated_target_distributions.append(row)
        
    # Aggregate results across seeds
    aggregated_results = []
    model_names = list(all_seed_results[0].keys())
    
    for model_name in model_names:
        mrr_vals = [r[model_name]["MRR"] for r in all_seed_results]
        hits3_vals = [r[model_name]["Hits@3"] for r in all_seed_results]
        hits5_vals = [r[model_name]["Hits@5"] for r in all_seed_results]
        f1_vals = [r[model_name]["F1"] for r in all_seed_results]
        mse_vals = [r[model_name]["MSE"] for r in all_seed_results]
        mae_vals = [r[model_name]["MAE"] for r in all_seed_results]
        ece_vals = [r[model_name]["ECE"] for r in all_seed_results]
        time_vals = [r[model_name]["Train Time (s)"] for r in all_seed_results]
        
        agg = {
            "Model": model_name,
            "Dataset": dataset_name,
            "Uncertainty": uncertainty_level,
            "MRR_mean": float(np.mean(mrr_vals)),
            "MRR_std": float(np.std(mrr_vals, ddof=1)) if len(mrr_vals) > 1 else 0.0,
            "Hits@3_mean": float(np.mean(hits3_vals)),
            "Hits@3_std": float(np.std(hits3_vals, ddof=1)) if len(hits3_vals) > 1 else 0.0,
            "Hits@5_mean": float(np.mean(hits5_vals)),
            "Hits@5_std": float(np.std(hits5_vals, ddof=1)) if len(hits5_vals) > 1 else 0.0,
            "F1_mean": float(np.mean(f1_vals)),
            "F1_std": float(np.std(f1_vals, ddof=1)) if len(f1_vals) > 1 else 0.0,
            "MSE_mean": float(np.mean(mse_vals)),
            "MSE_std": float(np.std(mse_vals, ddof=1)) if len(mse_vals) > 1 else 0.0,
            "MAE_mean": float(np.mean(mae_vals)),
            "MAE_std": float(np.std(mae_vals, ddof=1)) if len(mae_vals) > 1 else 0.0,
            "ECE_mean": float(np.mean(ece_vals)),
            "ECE_std": float(np.std(ece_vals, ddof=1)) if len(ece_vals) > 1 else 0.0,
            "Time_mean": float(np.mean(time_vals)),
            "Time_std": float(np.std(time_vals, ddof=1)) if len(time_vals) > 1 else 0.0,
        }
        
        print(f"Model: {model_name:40s} | MRR: {agg['MRR_mean']:.4f} ± {agg['MRR_std']:.4f} | ECE: {agg['ECE_mean']:.4f} ± {agg['ECE_std']:.4f}")
        aggregated_results.append(agg)
        
    return aggregated_results, aggregated_coverage, aggregated_target_distributions

def main():
    parser = argparse.ArgumentParser(description="Multi-seed comparison of TransE and RFM-TransE")
    parser.add_argument("--dataset", type=str, default="CoDEx-S", 
                        choices=["CoDEx-S", "CoDEx-M", "FB15k", "FB15k-237", "WN18RR", "all"],
                        help="Dataset name or 'all' to run on all datasets.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 2024],
                        help="List of random seeds to run multi-seed evaluation.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=positive_integer, default=1024, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate.")
    parser.add_argument("--embedding_dim", type=int, default=50, help="Embedding dimension size.")
    parser.add_argument("--margin", type=float, default=2.0, help="TransE ranking loss margin.")
    parser.add_argument("--p_norm", type=int, default=1, choices=[1, 2], help="Norm for distance.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Initial relation fuzziness sharpness alpha.")
    parser.add_argument("--lambda_fuzzy", type=float, default=0.1, help="Fuzzy loss scaling factor.")
    parser.add_argument("--fuzzy_bias", type=float, default=0.0, help="Bias for sigmoid membership function.")
    parser.add_argument("--uncertainty", type=str, default="medium", choices=["low", "medium", "high", "all"],
                        help="Uncertainty level or 'all' to run on low, medium, and high.")
    parser.add_argument(
        "--models", nargs="+",
        choices=["standard", "sigmoid", "exponential", "gaussian"],
        default=["standard", "sigmoid", "exponential", "gaussian"],
        help="Models to run. Use '--models standard sigmoid' for the final thesis comparison."
    )
    parser.add_argument("--train_fraction", type=split_fraction, default=1.0,
                        help="Fraction of the official training split to use (default: all).")
    parser.add_argument("--val_fraction", type=split_fraction, default=1.0,
                        help="Fraction of the official validation split to use (default: all).")
    parser.add_argument("--test_fraction", type=split_fraction, default=1.0,
                        help="Fraction of the official test split to use (default: all).")
    parser.add_argument("--train_sample_size", type=int, default=0,
                        help="Optional training cap after fraction selection. <= 0 means no cap.")
    parser.add_argument("--eval_sample_size", type=int, default=0,
                        help="Optional shared validation/test cap. <= 0 means no cap.")
    parser.add_argument("--val_sample_size", type=int, default=None,
                        help="Override the shared evaluation cap for validation only.")
    parser.add_argument("--test_sample_size", type=int, default=None,
                        help="Override the shared evaluation cap for test only.")
    parser.add_argument("--eval_batch_size", type=positive_integer, default=16,
                        help="Ranking query batch size; lower it to reduce memory, not coverage.")
    parser.add_argument("--output_file", type=str, default="benchmark_results_full.md",
                        help="Path to save the benchmark results markdown report.")
    
    args = parser.parse_args()
    
    # Select Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using Device: {device}")
    
    datasets = ["CoDEx-S", "CoDEx-M", "FB15k", "FB15k-237", "WN18RR"] if args.dataset == "all" else [args.dataset]
    uncertainties = ["low", "medium", "high"] if args.uncertainty == "all" else [args.uncertainty]
    
    all_results = []
    all_coverage_statistics = []
    all_target_distributions = []
    
    for dataset in datasets:
        for level in uncertainties:
            try:
                res, coverage, target_distributions = run_experiment(
                    dataset, level, args, device
                )
                all_results.extend(res)
                if not any(
                    existing["Dataset"] == dataset
                    for existing in all_coverage_statistics
                ):
                    all_coverage_statistics.append(coverage)
                all_target_distributions.extend(target_distributions)
            except Exception as e:
                print(f"Error executing experiment for {dataset} ({level} uncertainty): {e}")
                import traceback
                traceback.print_exc()
                
    # Compile and save Markdown Report
    if all_results:
        report = "# RFM-TransE Multi-Seed Benchmark Report\n\n"
        report += f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Device: {device}\n"
        report += f"Evaluated Seeds: {args.seeds} (Mean ± Standard Deviation)\n"
        report += f"Hyperparameters: epochs={args.epochs}, lr={args.lr}, dim={args.embedding_dim}, "
        report += f"margin={args.margin}, lambda_fuzzy={args.lambda_fuzzy}\n\n"
        report += f"Models: {args.models}.\n"
        report += "Protocol: original train/validation/test files are preserved; no resplitting.\n"
        report += f"Fractions (train/validation/test): {args.train_fraction}/{args.val_fraction}/{args.test_fraction}.\n"
        report += f"Caps: train={args.train_sample_size}, shared evaluation={args.eval_sample_size}, "
        report += f"validation override={args.val_sample_size}, test override={args.test_sample_size}. "
        report += "Nonpositive caps mean unlimited; fractions apply first.\n"
        report += f"Ranking query batch size: {args.eval_batch_size}.\n\n"

        report += "The standard TransE continuous reference uses the fixed post-hoc mapping "
        report += "$\\sigma(-0.1d)$, matching the thesis evaluation.\n\n"

        report += "## Training Coverage\n\n"
        report += "Coverage is measured against the complete training split before sampling. Entity and "
        report += "relation coverage show whether a small triple sample still represents the vocabulary of "
        report += "the original training graph. Sampled counts and percentages are reported as mean $\\pm$ "
        report += "sample standard deviation across seeds.\n\n"
        report += "| Dataset | Full Train Triples | Sampled Train Triples | Triple Coverage | Full Train Entities | Sampled Train Entities | Entity Coverage | Full Train Relations | Sampled Train Relations | Relation Coverage |\n"
        report += "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        for coverage in all_coverage_statistics:
            report += f"| {coverage['Dataset']} | "
            report += f"{coverage['Full Train Triples_mean']:.0f} | "
            report += f"{coverage['Sampled Train Triples_mean']:.1f} ± {coverage['Sampled Train Triples_std']:.1f} | "
            report += f"{coverage['Triple Coverage (%)_mean']:.2f}% ± {coverage['Triple Coverage (%)_std']:.2f}% | "
            report += f"{coverage['Full Train Entities_mean']:.0f} | "
            report += f"{coverage['Sampled Train Entities_mean']:.1f} ± {coverage['Sampled Train Entities_std']:.1f} | "
            report += f"{coverage['Entity Coverage (%)_mean']:.2f}% ± {coverage['Entity Coverage (%)_std']:.2f}% | "
            report += f"{coverage['Full Train Relations_mean']:.0f} | "
            report += f"{coverage['Sampled Train Relations_mean']:.1f} ± {coverage['Sampled Train Relations_std']:.1f} | "
            report += f"{coverage['Relation Coverage (%)_mean']:.2f}% ± {coverage['Relation Coverage (%)_std']:.2f}% |\n"

        report += "\n## Validation and Test Coverage\n\n"
        report += "Coverage is relative to each original split, not to the training graph. "
        report += "Full split use does not guarantee that every validation/test entity appears in training.\n\n"
        report += "| Dataset | Split | Full Triples | Used Triples | Triple Coverage | Entity Coverage | Relation Coverage |\n"
        report += "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |\n"
        for coverage in all_coverage_statistics:
            for name in ("Validation", "Test"):
                report += f"| {coverage['Dataset']} | {name} | "
                report += f"{coverage[f'{name} Full Train Triples_mean']:.0f} | "
                report += f"{coverage[f'{name} Sampled Train Triples_mean']:.1f} ± {coverage[f'{name} Sampled Train Triples_std']:.1f} | "
                for metric in ("Triple", "Entity", "Relation"):
                    key = f"{name} {metric} Coverage (%)"
                    report += f"{coverage[key + '_mean']:.2f}% ± {coverage[key + '_std']:.2f}% | "
                report += "\n"

        report += "\n## Soft Target Distribution Audit\n\n"
        report += "The table audits the fixed positive training targets and the fixed validation and test "
        report += "targets. Negative training triples are not listed because they are regenerated for every "
        report += "batch and epoch. The within-set SD describes variation among targets inside one run; the "
        report += "value after $\\pm$ describes variation of that statistic across seeds. Structurally adjusted "
        report += "is the percentage of targets for which Jaccard neighborhood overlap changed the regime's "
        report += "unadjusted base value.\n\n"
        report += "| Dataset | Uncertainty | Target Set | N per Seed | Mean Target | Within-Set SD | Min | Median | Max | Structurally Adjusted |\n"
        report += "| :--- | :---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        for target in all_target_distributions:
            report += f"| {target['Dataset']} | {target['Uncertainty'].upper()} | {target['Target Set']} | "
            report += f"{target['Count_mean']:.1f} ± {target['Count_std']:.1f} | "
            report += f"{target['Mean_mean']:.4f} ± {target['Mean_std']:.4f} | "
            report += f"{target['Within-Set SD_mean']:.4f} ± {target['Within-Set SD_std']:.4f} | "
            report += f"{target['Min_mean']:.4f} ± {target['Min_std']:.4f} | "
            report += f"{target['Median_mean']:.4f} ± {target['Median_std']:.4f} | "
            report += f"{target['Max_mean']:.4f} ± {target['Max_std']:.4f} | "
            report += f"{target['Structurally Adjusted (%)_mean']:.2f}% ± {target['Structurally Adjusted (%)_std']:.2f}% |\n"
        report += "\n"
        
        report += "## Performance Results (Mean ± Std Across Seeds)\n\n"
        report += "| Dataset | Model | Uncertainty | MRR | Hits@3 | Hits@5 | F1 Score | MSE | MAE | ECE | Train Time (s) |\n"
        report += "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        
        for res in all_results:
            report += f"| {res['Dataset']} | {res['Model']} | {res['Uncertainty'].upper()} | "
            report += f"{res['MRR_mean']:.4f} ± {res['MRR_std']:.4f} | {res['Hits@3_mean']:.4f} ± {res['Hits@3_std']:.4f} | {res['Hits@5_mean']:.4f} ± {res['Hits@5_std']:.4f} | "
            report += f"{res['F1_mean']:.4f} ± {res['F1_std']:.4f} | {res['MSE_mean']:.4f} ± {res['MSE_std']:.4f} | {res['MAE_mean']:.4f} ± {res['MAE_std']:.4f} | {res['ECE_mean']:.4f} ± {res['ECE_std']:.4f} | "
            report += f"{res['Time_mean']:.1f}s ± {res['Time_std']:.1f}s |\n"
            
        print("\n" + "="*100)
        print("BENCHMARK COMPLETED SUCCESSFULLY")
        print("="*100 + "\n")
        print(report)
        
        with open(args.output_file, "w") as f:
            f.write(report)
        print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
