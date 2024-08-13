use std::ops::{Index, IndexMut};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Shape {
    dims: (usize, usize, usize),
    data: Vec<bool>,
}

type Transform = (usize, usize, usize);

const TRANSFORMS: [Transform; 24] = [
    (0, 1, 2),
    (0, !1, !2),
    (!0, 1, !2),
    (!0, !1, 2),
    (0, 2, !1),
    (0, !2, 1),
    (!0, 2, 1),
    (!0, !2, !1),
    (1, 0, !2),
    (1, !0, 2),
    (!1, 0, 2),
    (!1, !0, !2),
    (1, 2, 0),
    (1, !2, !0),
    (!1, 2, !0),
    (!1, !2, 0),
    (2, 0, 1),
    (2, !0, !1),
    (!2, 0, !1),
    (!2, !0, 1),
    (2, 1, !0),
    (2, !1, 0),
    (!2, 1, 0),
    (!2, !1, !0),
];

fn get_index(dims: (usize, usize, usize), idx: usize) -> usize {
    if idx == 0 || idx == !0 {
        dims.0
    } else if idx == 1 || idx == !1 {
        dims.1
    } else if idx == 2 || idx == !2 {
        dims.2
    } else {
        panic!("Invalid index: {}", idx);
    }
}

fn get_index_offset(pos: (usize, usize, usize), dims: (usize, usize, usize), idx: usize) -> usize {
    if idx == 0 {
        pos.0
    } else if idx == !0 {
        dims.0 - pos.0 - 1
    } else if idx == 1 {
        pos.1
    } else if idx == !1 {
        dims.1 - pos.1 - 1
    } else if idx == 2 {
        pos.2
    } else if idx == !2 {
        dims.2 - pos.2 - 1
    } else {
        panic!("Invalid index: {}", idx);
    }
}

impl Shape {
    pub fn new(data: Vec<bool>, dims: (usize, usize, usize)) -> Shape {
        Self { data, dims }
    }

    pub fn dims(&self) -> (usize, usize, usize) {
        self.dims
    }

    pub fn from_array_2d(data: Vec<Vec<bool>>) -> Shape {
        let dims = (data.len(), data[0].len(), 1);
        let data = data.into_iter().flatten().collect();
        Self { data, dims }
    }

    pub fn from_array_3d(data: Vec<Vec<Vec<bool>>>) -> Shape {
        let dims = (data.len(), data[0].len(), data[0][0].len());
        let data = data.into_iter().flatten().flatten().collect();
        Self { data, dims }
    }

    fn apply_transform(&self, transform: Transform) -> Shape {
        let new_dims = (
            get_index(self.dims, transform.0),
            get_index(self.dims, transform.1),
            get_index(self.dims, transform.2),
        );
        let mut ret = Shape::new(vec![false; new_dims.0 * new_dims.1 * new_dims.2], new_dims);
        for i in 0..self.dims.0 {
            for j in 0..self.dims.1 {
                for k in 0..self.dims.2 {
                    let pos = (i, j, k);
                    let new_pos = (
                        get_index_offset(pos, self.dims, transform.0),
                        get_index_offset(pos, self.dims, transform.1),
                        get_index_offset(pos, self.dims, transform.2),
                    );
                    ret[new_pos] = self[pos];
                }
            }
        }
        ret
    }

    pub fn enumerate_transforms(&self) -> Vec<Shape> {
        let mut ret = vec![];
        for &transform in &TRANSFORMS {
            ret.push(self.apply_transform(transform));
        }
        ret.sort();
        ret.dedup();
        ret
    }

    pub fn origin(&self) -> (usize, usize, usize) {
        for i in 0..self.data.len() {
            if self.data[i] {
                let z = i % self.dims.2;
                let y = (i / self.dims.2) % self.dims.1;
                let x = i / (self.dims.1 * self.dims.2);
                return (x, y, z);
            }
        }

        panic!("empty shape");
    }
}

impl Index<(usize, usize, usize)> for Shape {
    type Output = bool;

    fn index(&self, index: (usize, usize, usize)) -> &Self::Output {
        let (x, y, z) = index;
        let (_wx, wy, wz) = self.dims;
        let index = x * wy * wz + y * wz + z;
        &self.data[index]
    }
}

impl IndexMut<(usize, usize, usize)> for Shape {
    fn index_mut(&mut self, index: (usize, usize, usize)) -> &mut Self::Output {
        let (x, y, z) = index;
        let (_wx, wy, wz) = self.dims;
        let index = x * wy * wz + y * wz + z;
        &mut self.data[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_constructor() {
        {
            let data = vec![true, false, true, false];
            let dims = (2, 2, 1);
            let shape = Shape::new(data.clone(), dims);
            assert_eq!(shape.data, data);
            assert_eq!(shape.dims, dims);
        }

        {
            let shape =
                Shape::from_array_3d(vec![vec![vec![true, false, true], vec![true, true, true]]]);
            assert_eq!(shape.data, vec![true, false, true, true, true, true]);
            assert_eq!(shape.dims, (1, 2, 3));
        }
    }

    #[test]
    fn test_transform() {
        {
            let mut shape = Shape::new(vec![true; 120], (3, 4, 5));
            shape[(0, 1, 2)] = false;

            let transformed = shape.apply_transform((!1, !2, 0));
            assert_eq!(transformed.dims, (4, 5, 3));

            for i in 0..4 {
                for j in 0..5 {
                    for k in 0..3 {
                        assert_eq!(transformed[(i, j, k)], (i, j, k) != (2, 2, 0));
                    }
                }
            }
        }
    }

    #[test]
    fn test_enumerate_transforms() {
        {
            let shape = Shape::from_array_3d(vec![vec![vec![true]]]);

            let transforms = shape.enumerate_transforms();

            assert_eq!(transforms, vec![shape]);
        }

        {
            let shape = Shape::from_array_3d(vec![vec![vec![true, true]]]);
            let transforms = shape.enumerate_transforms();

            assert_eq!(transforms.len(), 3);
        }

        {
            let shape = Shape::from_array_3d(vec![vec![vec![true, true], vec![true, false]]]);
            let transforms = shape.enumerate_transforms();

            assert_eq!(transforms.len(), 12);
        }

        {
            let shape =
                Shape::from_array_3d(vec![vec![vec![true, true, true], vec![true, false, false]]]);
            let transforms = shape.enumerate_transforms();

            assert_eq!(transforms.len(), 24);
        }
    }
}
