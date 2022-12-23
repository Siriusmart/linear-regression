use csv::Reader;
use std::{fs::File, io};

const LEARNING_RATE: f64 = 0.000000000012;

fn main() {
    println!("Enter path to csv or params:");

    let stdin = io::stdin();
    let mut input = String::new();
    stdin.read_line(&mut input).unwrap();

    if input.contains(",") {
        let two_lines = input.split("|").collect::<Vec<_>>();
        let weights = two_lines[1]
            .split(",")
            .map(|n| n.trim().parse::<f64>().unwrap())
            .collect::<Vec<_>>();

        println!("\n\nEnter values ({})", two_lines[0]);
        input.clear();
        stdin.read_line(&mut input).unwrap();

        let features = input
            .split(",")
            .map(|n| n.trim().parse::<f64>().unwrap())
            .collect::<Vec<_>>();

        if weights.len() != features.len() {
            panic!(
                "Expected {} features, found {}",
                weights.len(),
                features.len()
            );
        }

        let output = output(&weights, &features);

        println!("\n\nOutput: {output}");

        return;
    }

    let mut reader = Reader::from_path(input.trim()).unwrap();

    let records = get_records(&mut reader, usize::MAX);

    let mut features = get_feature_names(&mut reader);
    let target_index = features.len() - 1;
    features.pop();

    let data = get_data(&records, target_index);
    let targets = get_targets(&records, target_index);

    let mut weights = random_weight(features.len());
    let mut slopes = (0..features.len()).map(|_| fastrand::f64() - 0.5).collect();

    for _ in 0..60 {
        for index in 0..features.len() {
            let mut previous_cost = total_cost(&weights, &data, &targets);

            for _ in 0..150 {
                regression_single(
                    &mut weights,
                    &data,
                    &targets,
                    &mut slopes,
                    index,
                    &mut previous_cost,
                );
            }
        }
    }

    let accuracy = total_cost(&weights, &data, &targets) / data.len() as f64;
    println!("\n\nAccuracy after: {accuracy}");
    println!(
        "Results:\n\n{}|{}",
        features.join(","),
        weights
            .into_iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
}

// get functions
fn get_feature_names(reader: &mut Reader<File>) -> Vec<String> {
    reader
        .headers()
        .unwrap()
        .into_iter()
        .map(|s| s.to_string())
        .collect()
}

fn get_records(reader: &mut Reader<File>, take: usize) -> Vec<Vec<f64>> {
    reader
        .records()
        .take(take)
        .map(|row| {
            row.unwrap()
                .into_iter()
                .map(|cell| cell.parse().unwrap())
                .collect()
        })
        .collect()
}

fn get_data(records: &Vec<Vec<f64>>, target_index: usize) -> Vec<Vec<f64>> {
    records
        .iter()
        .map(|row| row[0..target_index - 1].to_vec())
        .collect::<Vec<_>>()
}

fn get_targets(records: &Vec<Vec<f64>>, target_index: usize) -> Vec<f64> {
    records.iter().map(|row| row[target_index]).collect()
}

// cost functions
fn total_cost(weights: &Vec<f64>, data: &Vec<Vec<f64>>, targets: &Vec<f64>) -> f64 {
    data.iter()
        .zip(targets.iter())
        .map(|(row, target)| single_cost(weights, row, *target))
        .sum()
}

fn single_cost(weights: &Vec<f64>, features: &Vec<f64>, target: f64) -> f64 {
    (output(weights, features) - target).abs()
}

fn output(weights: &Vec<f64>, features: &Vec<f64>) -> f64 {
    weights
        .iter()
        .zip(features.into_iter())
        .map(|(&weight, &cell)| weight * cell)
        .sum::<f64>()
}

// regression
fn regression_single(
    weights: &mut Vec<f64>,
    data: &Vec<Vec<f64>>,
    targets: &Vec<f64>,
    slopes: &mut Vec<f64>,
    weight_index: usize,
    previous_cost: &mut f64,
) {
    let previous_weight = weights[weight_index];
    let slope = slopes[weight_index];

    let weight_change = -slope * LEARNING_RATE;
    let new_weight = previous_weight + weight_change;

    weights[weight_index] = new_weight;

    let new_cost = total_cost(weights, data, targets);

    if new_weight == previous_weight {
        return;
    }

    slopes[weight_index] = (new_cost - *previous_cost) / (new_weight - previous_weight);

    // println!("Slope before: {slope} | Slope after: {}", slopes[weight_index]);
}

// weights
fn random_weight(length: usize) -> Vec<f64> {
    (0..length).map(|_| fastrand::f64()).collect()
}
