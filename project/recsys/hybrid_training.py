"""Fixed-budget train-only fitting shared by alternative hybrids."""

from .recbole_adapter import export_scores


def train_and_score(config, dataset, training, evaluation, directory, split_id, label):
    """Reuse Task 1.1 model factories/exporter with explicit interaction subsets."""
    from recbole.data.utils import create_samplers, get_dataloader
    from recbole.utils import get_model, get_trainer, init_seed

    directory.mkdir(parents=True, exist_ok=True)
    config["checkpoint_dir"] = str(directory)
    # Third partition is deliberately empty. Neither sampling nor training sees
    # outer validation/test interactions when producing OOF features.
    empty = dataset.copy(training.inter_feat[:0])
    samplers = create_samplers(config, dataset, [training, evaluation, empty])
    train_loader = get_dataloader(config, "train")(config, training, samplers[0], shuffle=True)
    eval_loader = get_dataloader(config, "valid")(config, evaluation, samplers[1], shuffle=False)
    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, training).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    try:
        trainer.fit(train_loader, valid_data=None, saved=True, verbose=False, show_progress=False)
        model.eval()
        return export_scores(model, dataset, eval_loader, split_id, "valid", label)
    finally:
        if hasattr(trainer, "tensorboard"):
            trainer.tensorboard.close()
