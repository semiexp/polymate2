use std::ops::{Add, Index, IndexMut, Mul, Not, Sub};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Coord(pub i32, pub i32, pub i32);

impl Coord {
    pub fn ge(&self, other: Coord) -> bool {
        self.0 >= other.0 && self.1 >= other.1 && self.2 >= other.2
    }

    fn cmp_lex(&self, other: Coord) -> std::cmp::Ordering {
        self.0
            .cmp(&other.0)
            .then(self.1.cmp(&other.1))
            .then(self.2.cmp(&other.2))
    }

    pub(crate) fn volume(&self) -> i32 {
        self.0 * self.1 * self.2
    }
}

impl Add for Coord {
    type Output = Coord;

    fn add(self, rhs: Coord) -> Self::Output {
        Coord(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2)
    }
}

impl Sub for Coord {
    type Output = Coord;

    fn sub(self, rhs: Coord) -> Self::Output {
        Coord(self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CubicGrid<T> {
    dims: Coord,
    pub(crate) data: Vec<T>,
}

impl<T: PartialOrd> PartialOrd for CubicGrid<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let c = self.dims.cmp_lex(other.dims);

        if c != std::cmp::Ordering::Equal {
            return Some(c);
        }

        self.data.partial_cmp(&other.data)
    }
}

impl<T: Ord> Ord for CubicGrid<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dims
            .cmp_lex(other.dims)
            .then_with(|| self.data.cmp(&other.data))
    }
}

impl<T> CubicGrid<T> {
    pub fn new(data: Vec<T>, dims: Coord) -> CubicGrid<T> {
        assert_eq!(data.len() as i32, dims.volume());
        Self { data, dims }
    }

    pub fn dims(&self) -> Coord {
        self.dims
    }

    pub fn from_array_2d(data: Vec<Vec<T>>) -> CubicGrid<T> {
        let dims = Coord(1, data.len() as i32, data[0].len() as i32);
        let data = data.into_iter().flatten().collect();
        CubicGrid::new(data, dims)
    }

    pub fn from_array_3d(data: Vec<Vec<Vec<T>>>) -> CubicGrid<T> {
        let dims = Coord(
            data.len() as i32,
            data[0].len() as i32,
            data[0][0].len() as i32,
        );
        let data = data.into_iter().flatten().flatten().collect();
        CubicGrid::new(data, dims)
    }

    pub(crate) fn apply_transform(&self, transform: Transform) -> CubicGrid<T>
    where
        T: Clone,
    {
        let new_dims = Coord(
            get_index(self.dims, transform.0),
            get_index(self.dims, transform.1),
            get_index(self.dims, transform.2),
        );

        let inv = !transform;
        let mut buf = vec![];
        for i in 0..new_dims.0 {
            for j in 0..new_dims.1 {
                for k in 0..new_dims.2 {
                    let pos = Coord(i, j, k);
                    let orig_pos = Coord(
                        get_index_offset(pos, new_dims, inv.0),
                        get_index_offset(pos, new_dims, inv.1),
                        get_index_offset(pos, new_dims, inv.2),
                    );
                    buf.push(self[orig_pos].clone());
                }
            }
        }

        CubicGrid {
            data: buf,
            dims: new_dims,
        }
    }
}

impl<T> Index<(usize, usize, usize)> for CubicGrid<T> {
    type Output = T;

    fn index(&self, index: (usize, usize, usize)) -> &Self::Output {
        assert!(
            (index.0 as i32) < self.dims.0
                && (index.1 as i32) < self.dims.1
                && (index.2 as i32) < self.dims.2
        );
        let (x, y, z) = index;
        let Coord(_wx, wy, wz) = self.dims;
        let index = x as i32 * wy * wz + y as i32 * wz + z as i32;
        &self.data[index as usize]
    }
}

impl<T> Index<Coord> for CubicGrid<T> {
    type Output = T;

    fn index(&self, index: Coord) -> &Self::Output {
        assert!(index.0 >= 0 && index.1 >= 0 && index.2 >= 0);
        self.index((index.0 as usize, index.1 as usize, index.2 as usize))
    }
}

impl<T> IndexMut<(usize, usize, usize)> for CubicGrid<T> {
    fn index_mut(&mut self, index: (usize, usize, usize)) -> &mut Self::Output {
        assert!(
            (index.0 as i32) < self.dims.0
                && (index.1 as i32) < self.dims.1
                && (index.2 as i32) < self.dims.2
        );
        let (x, y, z) = index;
        let Coord(_wx, wy, wz) = self.dims;
        let index = x as i32 * wy * wz + y as i32 * wz + z as i32;
        &mut self.data[index as usize]
    }
}

impl<T> IndexMut<Coord> for CubicGrid<T> {
    fn index_mut(&mut self, index: Coord) -> &mut Self::Output {
        assert!(index.0 >= 0 && index.1 >= 0 && index.2 >= 0);
        self.index_mut((index.0 as usize, index.1 as usize, index.2 as usize))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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

pub(crate) const TRANSFORMS: [Transform; 24] = enumerate_transforms(false);
pub(crate) const MIRROR_TRANSFORMS: [Transform; 24] = enumerate_transforms(true);

pub(crate) fn transform_bbox(
    top: Coord,
    vol: Coord,
    board_dims: Coord,
    transform: Transform,
) -> Coord {
    let get = |idx| {
        if idx == 0 {
            top.0
        } else if idx == !0 {
            board_dims.0 - top.0 - vol.0
        } else if idx == 1 {
            top.1
        } else if idx == !1 {
            board_dims.1 - top.1 - vol.1
        } else if idx == 2 {
            top.2
        } else if idx == !2 {
            board_dims.2 - top.2 - vol.2
        } else {
            panic!("Invalid index: {}", idx);
        }
    };
    Coord(get(transform.0), get(transform.1), get(transform.2))
}

fn get_index(dims: Coord, idx: usize) -> i32 {
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

fn get_index_offset(pos: Coord, dims: Coord, idx: usize) -> i32 {
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

    pub fn origin(&self) -> Coord {
        for i in 0..self.data.len() {
            if self.data[i] {
                let i = i as i32;
                let z = i % self.dims.2;
                let y = (i / self.dims.2) % self.dims.1;
                let x = i / (self.dims.1 * self.dims.2);
                return Coord(x as i32, y as i32, z as i32);
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

pub fn shape_from_string(s: &str) -> Shape {
    let lines = s
        .trim()
        .lines()
        .map(|line| line.trim_start().split(" ").collect::<Vec<_>>())
        .collect::<Vec<_>>();

    let num_layers = lines[0].len();
    let height = lines.len();
    let width = lines[0][0].len();

    let mut ret = Shape::new(
        vec![false; width * height * num_layers],
        Coord(num_layers as i32, height as i32, width as i32),
    );
    for y in 0..height {
        for z in 0..num_layers {
            assert_eq!(lines[y][z].len(), width);
            for (x, s) in lines[y][z].chars().enumerate() {
                ret[(z, y, x)] = s == '#';
            }
        }
    }

    ret
}

pub fn get_shape_by_name(name: char) -> Shape {
    let base_str = match name {
        'm' => "#",
        'd' => "##",
        'a' => "###",
        'b' => {
            "##
             #."
        }
        'i' => "####",
        'o' => {
            "##
             ##"
        }
        'l' => {
            "###
             #.."
        }
        't' => {
            "###
             .#."
        }
        'n' => {
            "##.
             .##"
        }
        'x' => {
            "#. ##
             .. #."
        }
        'y' => {
            ".. ##
             #. #."
        }
        'z' => {
            ".# ##
             .. #."
        }
        'F' => {
            ".##
             ##.
             .#."
        }
        'I' => "#####",
        'L' => {
            "####
             #..."
        }
        'N' => {
            ".###
             ##.."
        }
        'P' => {
            "###
             ##."
        }
        'T' => {
            "###
             .#.
             .#."
        }
        'U' => {
            "###
             #.#"
        }
        'V' => {
            "###
             #..
             #.."
        }
        'W' => {
            ".##
             ##.
             #.."
        }
        'X' => {
            ".#.
             ###
             .#."
        }
        'Y' => {
            "####
             .#.."
        }
        'Z' => {
            ".##
             .#.
             ##."
        }
        _ => panic!("Unknown shape name: {}", name),
    };

    shape_from_string(base_str)
}

pub fn get_shapes_by_names(names: &str) -> (Vec<Shape>, Vec<u32>) {
    let mut shapes = vec![];
    let mut counts = vec![];

    for c in names.chars() {
        let shape = get_shape_by_name(c);
        let idx = shapes.iter().position(|s| *s == shape);
        if let Some(idx) = idx {
            counts[idx] += 1;
        } else {
            shapes.push(shape);
            counts.push(1);
        }
    }

    (shapes, counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_constructor() {
        {
            let data = vec![true, false, true, false];
            let dims = Coord(2, 2, 1);
            let shape = Shape::new(data.clone(), dims);
            assert_eq!(shape.data, data);
            assert_eq!(shape.dims, Coord(2, 2, 1));
        }

        {
            let shape =
                Shape::from_array_3d(vec![vec![vec![true, false, true], vec![true, true, true]]]);
            assert_eq!(shape.data, vec![true, false, true, true, true, true]);
            assert_eq!(shape.dims, Coord(1, 2, 3));
        }
    }

    #[test]
    fn test_transform() {
        {
            let mut shape = Shape::new(vec![true; 60], Coord(3, 4, 5));
            shape[(0, 1, 2)] = false;

            let transformed = shape.apply_transform(Transform(!1, !2, 0));
            assert_eq!(transformed.dims, Coord(4, 5, 3));

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
    fn test_transform_bbox() {
        for i in 0..60i32 {
            let x = i / 20;
            let y = i / 5 % 4;
            let z = i % 5;

            let mut shape = Shape::new(vec![false; 60], Coord(3, 4, 5));
            shape[Coord(x, y, z)] = true;

            for transform in &TRANSFORMS {
                let transformed = shape.apply_transform(*transform);

                let c = transform_bbox(Coord(x, y, z), Coord(1, 1, 1), Coord(3, 4, 5), *transform);
                assert_eq!(transformed[c], true);
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

        let asymmetry_shape = shape_from_string(
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

    #[test]
    fn test_shape_from_string() {
        // 2D shape
        {
            let expected =
                Shape::from_array_2d(vec![vec![true, true, true], vec![true, true, false]]);
            let actual = shape_from_string(
                "###
                 ##.",
            );

            assert_eq!(actual, expected);
        }

        // 3D shape
        {
            let expected = Shape::from_array_3d(vec![
                vec![vec![false, true], vec![false, false]],
                vec![vec![true, true], vec![true, false]],
            ]);
            let actual = shape_from_string(
                ".# ##
                 .. #.",
            );

            assert_eq!(actual, expected);
        }
    }
}
