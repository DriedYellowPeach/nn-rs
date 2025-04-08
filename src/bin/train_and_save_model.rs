use std::path::Path;

use nn_rs::prelude::*;

fn main() {
    env_logger::init();
    let args = std::env::args().collect::<Vec<_>>();
    log::info!("Starting training...");
    let saved_path = Path::new(args[1].as_str());

    if let Some(parent) = saved_path.parent() {
        if !parent.exists() {
            log::error!("Path not exist: {:?}", parent);
            return;
        }
    } else {
        log::error!("No parent path found for: {:?}", saved_path);
        return;
    }

    log::info!("Saving model to {}", args[1]);

    let mut nn = NNBuilder::new_from_arch(&[28 * 28, 100, 32, 32, 10])
        .with_activator(ActivatorType::Sigmoid)
        .with_learn_rate(0.5)
        .with_epochs(1000)
        .with_mini_batch_size(30)
        .with_target_accuracy(0.97)
        .build()
        .unwrap();

    let (trn_size, test_size) = (50000, 10000);

    let data = DataSet::from_mnist("data/", trn_size, test_size).unwrap();

    nn.train(data);
    nn.save(saved_path).unwrap();
}
