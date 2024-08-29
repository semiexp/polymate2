use std::ops::{Index, IndexMut};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CubicGrid<T> {
    dims: (usize, usize, usize),
    pub(crate) data: Vec<T>,
}

impl<T> CubicGrid<T> {
    pub fn new(data: Vec<T>, dims: (usize, usize, usize)) -> CubicGrid<T> {
        Self { data, dims }
    }

    pub fn dims(&self) -> (usize, usize, usize) {
        self.dims
    }

    pub fn from_array_2d(data: Vec<Vec<T>>) -> CubicGrid<T> {
        let dims = (1, data.len(), data[0].len());
        let data = data.into_iter().flatten().collect();
        Self { data, dims }
    }

    pub fn from_array_3d(data: Vec<Vec<Vec<T>>>) -> CubicGrid<T> {
        let dims = (data.len(), data[0].len(), data[0][0].len());
        let data = data.into_iter().flatten().flatten().collect();
        Self { data, dims }
    }

    pub(crate) fn apply_transform(&self, transform: Transform) -> CubicGrid<T>
    where
        T: Clone,
    {
        let new_dims = (
            get_index(self.dims, transform.0),
            get_index(self.dims, transform.1),
            get_index(self.dims, transform.2),
        );

        let inv = inverse(transform);
        let mut buf = vec![];
        for i in 0..new_dims.0 {
            for j in 0..new_dims.1 {
                for k in 0..new_dims.2 {
                    let pos = (i, j, k);
                    let orig_pos = (
                        get_index_offset(pos, new_dims, inv.0),
                        get_index_offset(pos, new_dims, inv.1),
                        get_index_offset(pos, new_dims, inv.2),
                    );
                    buf.push(self[orig_pos].clone());
                }
            }
        }

        CubicGrid::new(buf, new_dims)
    }
}

impl<T> Index<(usize, usize, usize)> for CubicGrid<T> {
    type Output = T;

    fn index(&self, index: (usize, usize, usize)) -> &Self::Output {
        assert!(index.0 < self.dims.0 && index.1 < self.dims.1 && index.2 < self.dims.2);
        let (x, y, z) = index;
        let (_wx, wy, wz) = self.dims;
        let index = x * wy * wz + y * wz + z;
        &self.data[index]
    }
}

impl<T> IndexMut<(usize, usize, usize)> for CubicGrid<T> {
    fn index_mut(&mut self, index: (usize, usize, usize)) -> &mut Self::Output {
        assert!(index.0 < self.dims.0 && index.1 < self.dims.1 && index.2 < self.dims.2);
        let (x, y, z) = index;
        let (_wx, wy, wz) = self.dims;
        let index = x * wy * wz + y * wz + z;
        &mut self.data[index]
    }
}

pub(crate) type Transform = (usize, usize, usize);

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

fn inverse(transform: Transform) -> Transform {
    let mut ret = (0, 0, 0);

    if transform.0 == 0 {
        ret.0 = 0;
    } else if transform.0 == !0 {
        ret.0 = !0;
    } else if transform.0 == 1 {
        ret.1 = 0;
    } else if transform.0 == !1 {
        ret.1 = !0;
    } else if transform.0 == 2 {
        ret.2 = 0;
    } else if transform.0 == !2 {
        ret.2 = !0;
    } else {
        panic!("Invalid index: {}", transform.0);
    }

    if transform.1 == 0 {
        ret.0 = 1;
    } else if transform.1 == !0 {
        ret.0 = !1;
    } else if transform.1 == 1 {
        ret.1 = 1;
    } else if transform.1 == !1 {
        ret.1 = !1;
    } else if transform.1 == 2 {
        ret.2 = 1;
    } else if transform.1 == !2 {
        ret.2 = !1;
    } else {
        panic!("Invalid index: {}", transform.1);
    }

    if transform.2 == 0 {
        ret.0 = 2;
    } else if transform.2 == !0 {
        ret.0 = !2;
    } else if transform.2 == 1 {
        ret.1 = 2;
    } else if transform.2 == !1 {
        ret.1 = !2;
    } else if transform.2 == 2 {
        ret.2 = 2;
    } else if transform.2 == !2 {
        ret.2 = !2;
    } else {
        panic!("Invalid index: {}", transform.2);
    }

    ret
}

pub type Shape = CubicGrid<bool>;

impl Shape {
    pub fn enumerate_transforms(&self) -> Vec<Shape> {
        let mut ret = vec![];
        for &transform in &TRANSFORMS {
            ret.push(self.apply_transform(transform));
        }
        ret.sort();
        ret.dedup();
        ret
    }

    pub(crate) fn compute_symmetry(&self) -> Vec<Transform> {
        let mut ret = vec![];
        for tr in &TRANSFORMS {
            let transformed = self.apply_transform(*tr);
            if transformed == *self {
                ret.push(*tr);
            }
        }
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

    pub fn normalize(&self) -> Shape {
        self.enumerate_transforms().into_iter().min().unwrap()
    }
}

pub fn dedup_shapes(shapes: &[Shape]) -> Vec<Vec<usize>> {
    let mut normalized_shapes = shapes
        .iter()
        .enumerate()
        .map(|(i, s)| (s.normalize(), i))
        .collect::<Vec<_>>();
    normalized_shapes.sort();

    let mut ret = vec![];
    let mut i = 0;
    while i < normalized_shapes.len() {
        let mut group = vec![];
        let mut j = i;
        while j < normalized_shapes.len() && normalized_shapes[j].0 == normalized_shapes[i].0 {
            group.push(normalized_shapes[j].1);
            j += 1;
        }
        ret.push(group);
        i = j;
    }

    ret
}

pub type Answer = CubicGrid<Option<(usize, usize)>>;

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
