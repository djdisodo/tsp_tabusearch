#![feature(iter_array_chunks)]

use std::fs::File;
use rand::distr::Distribution;
use rand::thread_rng;
use untitled4::TspProblem;
//문제 생성
fn main() {
    let json = File::create_new("problem.json").unwrap();
    let count = 200;
    let size = [1000u32, 1000u32];
    let rand = rand::distr::Uniform::new(0u32, 1000).unwrap();
    let nodes: Vec<[i32; 2]> = rand.sample_iter(thread_rng()).array_chunks().take(count).map(|x| x.map(|x| x as i32)).collect();
    serde_json::to_writer(json, &TspProblem {
        size,
        nodes,
    }).unwrap();
}