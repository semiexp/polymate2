use std::ops::{Index, IndexMut, Mul, Not};

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

        let inv = !transform;
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

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Transform(usize, usize, usize);

const fn enumerate_transforms(parity: bool) -> [Transform; 24] {
    const PERMUTATIONS: [(Transform, bool); 6] = [
        (Transform(0, 1, 2), false),
        (Transform(0, 2, 1), true),
        (Transform(1, 0, 2), true),
        (Transform(1, 2, 0), false),
        (Transform(2, 0, 1), false),
        (Transform(2, 1, 0), true),
    ];

    let mut ret = [Transform(0, 0, 0); 24];
    let mut index = 0;

    let mut i = 0;
    while i < 48 {
        let mut p = PERMUTATIONS[i / 8].1;
        p ^= (i / 4) % 2 == 1;
        p ^= (i / 2) % 2 == 1;
        p ^= i % 2 == 1;

        if p == parity {
            let perm = PERMUTATIONS[i / 8].0;

            ret[index] = Transform(
                if (i / 4) % 2 == 1 { !perm.0 } else { perm.0 },
                if (i / 2) % 2 == 1 { !perm.1 } else { perm.1 },
                if i % 2 == 1 { !perm.2 } else { perm.2 },
            );
            index += 1;
        }

        i += 1;
    }

    ret
}

const TRANSFORMS: [Transform; 24] = enumerate_transforms(false);
const MIRROR_TRANSFORMS: [Transform; 24] = enumerate_transforms(true);

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

impl Not for Transform {
    type Output = Transform;

    fn not(self) -> Self::Output {
        let mut ret = Transform(0, 0, 0);

        let mut apply = |val, i| {
            if val == 0 {
                ret.0 = i;
            } else if val == !0 {
                ret.0 = !i;
            } else if val == 1 {
                ret.1 = i;
            } else if val == !1 {
                ret.1 = !i;
            } else if val == 2 {
                ret.2 = i;
            } else if val == !2 {
                ret.2 = !i;
            } else {
                panic!("Invalid index: {}", val);
            }
        };

        apply(self.0, 0);
        apply(self.1, 1);
        apply(self.2, 2);

        ret
    }
}

impl Mul<Transform> for Transform {
    // Composition of transformations
    // (f * g)(x) = f(g(x))
    type Output = Transform;

    fn mul(self, rhs: Transform) -> Self::Output {
        let get = |i| {
            if i == 0 {
                rhs.0
            } else if i == !0 {
                !rhs.0
            } else if i == 1 {
                rhs.1
            } else if i == !1 {
                !rhs.1
            } else if i == 2 {
                rhs.2
            } else if i == !2 {
                !rhs.2
            } else {
                panic!("Invalid index: {}", i);
            }
        };

        Transform(get(self.0), get(self.1), get(self.2))
    }
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

    pub fn mirror(&self) -> Shape {
        self.apply_transform(Transform(!0, 1, 2))
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

    pub(crate) fn compute_mirroring_symmetry(&self) -> Vec<Transform> {
        let mut ret = vec![];
        for tr in &MIRROR_TRANSFORMS {
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

            let transformed = shape.apply_transform(Transform(!1, !2, 0));
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
    fn test_transform_composition() {
        let all_transforms = TRANSFORMS
            .iter()
            .chain(MIRROR_TRANSFORMS.iter())
            .copied()
            .collect::<Vec<_>>();

        let asymmetry_shape = crate::utils::tests::shape_from_string(
            ".##
             ##.
             .#.",
        );

        for i in 0..48 {
            {
                let f = all_transforms[i];
                let g = !f;

                let actual = asymmetry_shape.apply_transform(f).apply_transform(g);
                assert_eq!(actual, asymmetry_shape);

                let actual = asymmetry_shape.apply_transform(g).apply_transform(f);
                assert_eq!(actual, asymmetry_shape);
            }

            for j in 0..48 {
                let f = all_transforms[i];
                let g = all_transforms[j];
                let fg = f * g;

                let expected = asymmetry_shape.apply_transform(g).apply_transform(f);
                let actual = asymmetry_shape.apply_transform(fg);

                assert_eq!(actual, expected);
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
