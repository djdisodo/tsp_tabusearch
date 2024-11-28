#![feature(iterator_try_collect)]

use std::fs::File;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread::{sleep, spawn};
use std::time::Duration;
use arraydeque::{ArrayDeque, Wrapping};
use arraydeque::behavior::Behavior;
use gfx_graphics::GfxGraphics;
use ndarray::Array2;
use piston_window::{Context, EventLoop, EventSettings, PistonWindow, WindowSettings};
use plotters::prelude::*;
use plotters::coord::types::RangedCoordf32;
use plotters_piston::{draw_piston_window, PistonBackend};
use rand::rngs::ThreadRng;
use rand::seq::index::sample;
use rand::seq::SliceRandom;
use rand::thread_rng;
use untitled4::{NeighbourGenerator, NeighbourSolution, ProblemIndexMove, TspProblem};


pub struct TspProblemSolver {
    tsp_problem: Arc<TspProblem>,
    distance: Array2<f32>,
    tabu_list: ArrayDeque<ProblemIndexMove, 100, Wrapping>,
    best_solution: (Vec<usize>, f32),
    rng: ThreadRng
}

impl TspProblemSolver {
    pub fn new(problem: Arc<TspProblem>, mut rng: ThreadRng) -> Result<Self, anyhow::Error> {
        let mut distance_matrix = Array2::<f32>::zeros([problem.nodes.len(); 2]);
        for i in 0..problem.nodes.len() {
            for j in 0..problem.nodes.len() {
                let diff = [
                    problem.nodes[i][0] - problem.nodes[j][0],
                    problem.nodes[i][1] - problem.nodes[j][1]
                ];
                let diff_squard = diff.map(|x| x.pow(2));
                let distance = f32::sqrt((diff_squard[0] + diff_squard[1]) as f32);

                distance_matrix[[i, j]] = distance;
            }
        }
        let mut initial_solution: Vec<usize> = (0..).take(problem.nodes.len()).collect();
        initial_solution.shuffle(&mut rng);
        let mut solver = TspProblemSolver {
            tsp_problem: problem,
            distance: distance_matrix,
            tabu_list: Default::default(),
            best_solution: (initial_solution, f32::MAX),
            rng: thread_rng()
        };
        let distance = solver.evaluate_solution(&mut solver.best_solution.0.iter().map(|x| *x)).unwrap();
        solver.best_solution.1 = distance;
        Ok(solver)

    }

    pub fn evaluate_solution(&self, solution: &mut impl Iterator<Item=usize>) -> Option<f32> {
        let mut current = solution.next()?;
        let mut distance = 0f32;
        for next in solution {
            distance += self.distance[[current, next]];
            current = next;
        }
        Some(distance)
    }
}

fn solve(sender: Arc<Mutex<Vec<usize>>>, problem: Arc<TspProblem>) -> Result<(), anyhow::Error> {
    let mut solver = TspProblemSolver::new(problem.clone(), thread_rng())?;
    // 100만번 반복
    let mut current_solution = solver.best_solution.clone();
    for iterations in 0..1000000 {
        let mut best_move_non_tabu = (NeighbourSolution::dummy(&current_solution.0), f32::MAX);
        let mut best_move_tabu: Option<(NeighbourSolution, f32)> = None;
        let neighbours = NeighbourGenerator::new(&current_solution.0);

        for neighbour in neighbours {
            let distance = solver.evaluate_solution(&mut neighbour.into_iter()).unwrap();

            if best_move_non_tabu.1 > distance && !solver.tabu_list.contains(&neighbour.problem_index_move()) {
                best_move_non_tabu = (neighbour, distance);
            }
            if best_move_tabu.map(|x| x.1).unwrap_or(solver.best_solution.1) > distance {
                best_move_tabu = Some((neighbour, distance))
            }
        }


        if let Some((neighbour, distance)) = best_move_tabu {
            solver.tabu_list.push_front(neighbour.problem_index_move());
            current_solution = (neighbour.into_iter().collect(), distance);
            solver.best_solution = current_solution.clone();
        } else {
            let (neighbour, distance) = best_move_non_tabu;
            solver.tabu_list.push_front(neighbour.problem_index_move());
            current_solution = (neighbour.into_iter().collect(), distance);
        }

        if iterations % 100 == 0 {
            println!("bestsol {:.09}", solver.best_solution.1);
            *sender.lock().unwrap() = current_solution.0.clone()
        }
    }

    return Ok(());
}

fn main() -> Result<(), anyhow::Error> {
    //문제 로드
    let problem: Arc<TspProblem> = Arc::new(serde_json::from_reader(File::open("problem.json")?)?);

    let problem_ref = problem.clone();
    let sender: Arc<Mutex<Vec<usize>>> = Default::default();
    let receiver = sender.clone();

    spawn(|| {
        solve(sender, problem)
    });

    draw(receiver, problem_ref)



}








fn draw(channel: Arc<Mutex<Vec<usize>>>, problem: Arc<TspProblem>) -> Result<(), anyhow::Error> {



    let mut window: PistonWindow = WindowSettings::new("example", [1000, 1020]).build().ok().unwrap();
    window.events.set_event_settings(EventSettings {
        max_fps: 60,
        lazy: true,
        ..Default::default()
    });
    while let Some(e) = draw_piston_window(&mut window, |root| {
        let mut root = root.into_drawing_area();
        root.fill(&RGBColor(240, 200, 200))?;

        let root = root.apply_coord_spec(Cartesian2d::<RangedCoordf32, RangedCoordf32>::new(
            0f32..1000f32,
            0f32..1000f32,
            (0..1000, 0..1000),
        ));

        let dot_and_label = |x: f32, y: f32| {
            return EmptyElement::at((x, y))
                + Circle::new((0, 0), 3, ShapeStyle::from(&BLACK).filled())
                + Text::new(
                format!("({:.2},{:.2})", x, y),
                (10, 0),
                ("sans-serif", 15.0).into_font(),
            );
        };


        let solution = channel.lock().unwrap();
        let mut iter = solution.iter();
        //let i = iter.next().unwrap();

        //root.draw(&dot_and_label(prev[0] as f32, prev[1] as f32))?;
        for i in iter {
            let mut prev = problem.nodes[*i];
            root.draw(&dot_and_label(prev[0] as f32, prev[1] as f32))?;
        }
        let coords: Vec<(f32, f32)> = solution.iter().map(|x| problem.nodes[*x]).map(|x| (x[0] as f32, x[1] as f32)).collect();
        root.draw(&PathElement::new(coords, ShapeStyle::from(&BLACK).filled()))?;
        root.present()?;
        //
        Ok(())
    }) {
    }
    Ok(())

}