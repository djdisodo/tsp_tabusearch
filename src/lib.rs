use serde_derive::*;
#[derive(Serialize, Deserialize)]
pub struct TspProblem {
    pub size: [u32; 2],
    pub nodes: Vec<[i32; 2]>
}


// 이동 저장하는 배열, 정렬됨을 보장
#[derive(Hash, Eq, PartialEq)]
pub struct ProblemIndexMove([usize; 2]);

impl ProblemIndexMove {
    pub fn new(mut data: [usize; 2]) -> Self {
        data.sort_unstable();
        Self(data)
    }
}

#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub struct SolutionIndexMove([usize; 2]);

impl SolutionIndexMove {
    pub fn new(mut data: [usize; 2]) -> Self {
        data.sort_unstable();
        Self(data)
    }
}

//이웃 해를 새로운 메모리 할당없이 사용할때
#[derive(Copy, Clone)]
pub struct NeighbourSolution<'a> {
    parent: &'a [usize],
    mov: SolutionIndexMove
}

impl<'a> NeighbourSolution<'a> {

    pub fn dummy(parent: &'a [usize]) -> Self {
        Self {
            parent,
            mov: SolutionIndexMove([0; 2])
        }
    }
    pub fn problem_index_move(&self) -> ProblemIndexMove {
        let mov = [self.parent[self.mov.0[0]], self.parent[self.mov.0[1]]];
        ProblemIndexMove::new(mov)
    }
}

pub struct NeighbourSolutionIterator<'a> {
    solution: &'a NeighbourSolution<'a>,
    cursor: usize
}

impl<'a> Iterator for NeighbourSolutionIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let mut i = self.cursor;
        if (self.solution.mov.0[0]..=self.solution.mov.0[1]).contains(&i) {
            i -= self.solution.mov.0[0];
            i = self.solution.mov.0[1] - i
        }
        if self.solution.parent.len() == i {
            None
        } else {
            self.cursor += 1;
            Some(self.solution.parent[i])
        }
    }
}

impl<'a> IntoIterator for &'a NeighbourSolution<'a> {
    type Item = usize;
    type IntoIter = NeighbourSolutionIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        NeighbourSolutionIterator {
            solution: self,
            cursor: 0
        }
    }
}

//이웃해를 생성하는 타입
pub struct NeighbourGenerator<'a> {
    parent: &'a [usize],
    index_iterator: Box<dyn Iterator<Item=[usize;2]>>
}

impl<'a> NeighbourGenerator<'a> {
    pub fn new(parent: &'a [usize]) -> Self {
        let parent_len = parent.len();
        Self {
            parent,
            index_iterator: Box::new(
                (0..parent_len).map(
                    move |x| ((x+1)..parent_len).map(
                        move |y| [x, y]
                    )
                ).flatten()
            )
        }
    }
}

impl<'a> Iterator for NeighbourGenerator<'a> {
    type Item = NeighbourSolution<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.index_iterator.next().map(|mov| {
            NeighbourSolution {
                parent: self.parent,
                mov: SolutionIndexMove::new(mov)
            }
        })
    }
}