use crate::Point;

use num_traits::{AsPrimitive, Num, One, Signed, Zero};
use serde::{Deserialize, Serialize};
use std::fmt;

/// A simple line segment from point `(x1, y1)` to `(x2, y1)`.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(
    bound(serialize = "T: Clone + Serialize"),
    into = "[T; 4]",
    from = "[T; 4]"
)]
pub struct Line<T> {
    pub x1: T,
    pub y1: T,
    pub x2: T,
    pub y2: T,
}

impl<T> Line<T> {
    /// Creates a new line segment.
    pub fn new(x1: T, y1: T, x2: T, y2: T) -> Self {
        Self { x1, y1, x2, y2 }
    }

    /// Creates a new line segment between the given points.
    pub fn from_points(a: Point<T>, b: Point<T>) -> Self {
        (a, b).into()
    }

    /// Returns true if the ending point has greater-or-equal x-coordinate as the beginning.
    pub fn is_x_sorted(&self) -> bool
    where
        T: PartialOrd,
    {
        self.x1 <= self.x2
    }

    /// Returns true if the ending point has greater-or-equal y-coordinate as the beginning.
    pub fn is_y_sorted(&self) -> bool
    where
        T: PartialOrd,
    {
        self.y1 <= self.y2
    }

    /// Maps a function to all of the coordinates.
    #[must_use]
    pub fn map<F, U>(self, f: F) -> Line<U>
    where
        F: Fn(T) -> U,
    {
        Line {
            x1: f(self.x1),
            y1: f(self.y1),
            x2: f(self.x2),
            y2: f(self.y2),
        }
    }

    /// Maps a function to both of the x coordinates.
    #[must_use]
    pub fn map_x<F>(self, f: F) -> Self
    where
        F: Fn(T) -> T,
    {
        Line {
            x1: f(self.x1),
            y1: self.y1,
            x2: f(self.x2),
            y2: self.y2,
        }
    }

    /// Maps a function to both of the y coordinates.
    #[must_use]
    pub fn map_y<F>(self, f: F) -> Self
    where
        F: Fn(T) -> T,
    {
        Line {
            x1: self.x1,
            y1: f(self.y1),
            x2: self.x2,
            y2: f(self.y2),
        }
    }

    /// Maps a function to the beginning and end point.
    #[must_use]
    pub fn map_points<F, U>(self, f: F) -> Line<U>
    where
        F: Fn(Point<T>) -> Point<U>,
    {
        let Line { x1, y1, x2, y2 } = self;
        Line::from_points(f((x1, y1).into()), f((x2, y2).into()))
    }

    /// Maps a function to the beginning point.
    #[must_use]
    pub fn map_begin<F>(self, f: F) -> Self
    where
        F: Fn(T) -> T,
    {
        Line {
            x1: f(self.x1),
            y1: f(self.y1),
            x2: self.x2,
            y2: self.y2,
        }
    }

    /// Maps a function to the ending point.
    #[must_use]
    pub fn map_end<F>(self, f: F) -> Self
    where
        F: Fn(T) -> T,
    {
        Line {
            x1: self.x1,
            y1: self.y1,
            x2: f(self.x2),
            y2: f(self.y2),
        }
    }
}

impl<T: Clone + Num + PartialOrd> Line<T> {
    /// Returns the first point `(x1, y1)`.
    pub fn begin(&self) -> Point<T> {
        [self.x1.clone(), self.y1.clone()].into()
    }

    /// Returns the end point `(x2, y2)`.
    pub fn end(&self) -> Point<T> {
        [self.x2.clone(), self.y2.clone()].into()
    }

    /// Returns the midpoint of this line.
    pub fn mid(&self) -> Point<T> {
        let two = || T::one() + T::one();
        let two = || Point::new(two(), two());
        (self.begin() + self.end()) / two()
    }

    /// Returns `end - begin`.
    pub fn vector(&self) -> Point<T> {
        self.end() - self.begin()
    }

    /// Reverses this line to go from `(x2, y2)` to `(x1, y1)`.
    pub fn rev(&self) -> Line<T> {
        let Self { x1, y1, x2, y2 } = self.clone();
        Self::new(x2, y2, x1, y1)
    }

    /// Returns a line with the beginning point having the lesser x-coordinate.
    ///
    /// If the x coordinate of both points is the same, the line is returned unchanged.
    pub fn sort_x(&self) -> Line<T> {
        if self.is_x_sorted() {
            self.clone()
        } else {
            self.rev()
        }
    }

    /// Returns a line with the beginning point having the lesser y-coordinate.
    ///
    /// If the y coordinate of both points is the same, the line is returned unchanged.
    pub fn sort_y(&self) -> Line<T> {
        if self.is_y_sorted() {
            self.clone()
        } else {
            self.rev()
        }
    }

    /// Linear interpolation between the two endpoints of this line segment.
    ///
    /// If `t` is 0 then the beginning is returned, if it's 1 then the endpoint is returned
    /// (with some possible floating point inaccuracy).
    ///
    /// Anything between 0 and 1 is a point within this line segment, 0.5 being the middle.
    /// If `t` is less than 0 or bigger than 1 then a point outside the line segment is returned.
    pub fn lerp(&self, t: f32) -> Point<f32>
    where
        T: AsPrimitive<f32>,
    {
        let (x1, y1) = self.begin().into();
        let (dx, dy) = self.vector().into();
        Point {
            x: x1.as_() + dx.as_() * t,
            y: y1.as_() + dy.as_() * t,
        }
    }

    /// Returns the length of this line in the euclid space.
    ///
    /// Same as `line.vector().distance_euclid()`.
    pub fn length_euclid(&self) -> f32
    where
        T: AsPrimitive<f32>,
    {
        self.vector().distance_euclid()
    }

    /// Returns the taxicab/manhattan length of this line.
    ///
    /// Same as `line.vector().distance_taxi()`
    pub fn length_taxi(&self) -> T
    where
        T: Signed,
    {
        self.vector().distance_taxi()
    }

    /// Returns the Chebyshev/king's move length of this line.
    ///
    /// Same as `line.vector().distance_king()`.
    pub fn length_king(&self) -> T
    where
        T: Signed,
    {
        self.vector().distance_king()
    }

    /// Returns an iterator over the points in this line using the Bresenham algorithm.
    ///
    /// The returned points will be on a (1,1)-grid and the returned order will be arbitrary.
    pub fn bresenham(&self) -> BresenhamIter<T>
    where
        T: Signed,
    {
        let Line { x1, y1, x2, y2 } = {
            let Line { x1, y1, x2, y2 } = self.clone();
            let dx = x2 - x1;
            let dy = y2 - y1;
            // flip if needed
            if dx + dy > T::zero() {
                self.clone()
            } else {
                self.rev()
            }
        };
        let dx = x2.clone() - x1.clone();
        let dy = y2.clone() - y1.clone();

        let u_is_x = dx >= dy;
        let (u, v, u_end, du, dv) = if u_is_x {
            (x1, y1, x2, dx, dy)
        } else {
            (y1, x1, y2, dy, dx)
        };
        let v_increment = if dv >= T::zero() { T::one() } else { -T::one() };
        let du = du.abs();
        let dv = dv.abs();

        BresenhamIter {
            u_is_x,
            v_increment,
            u,
            v,
            u_end,
            du,
            dv,
            err: T::zero(),
        }
    }
}

pub struct BresenhamIter<T: PartialOrd + One + Zero + Signed> {
    u_is_x: bool,
    u: T,
    v: T,
    u_end: T,
    v_increment: T,
    du: T,
    dv: T,
    err: T,
}

impl<T> Iterator for BresenhamIter<T>
where
    T: Clone + PartialOrd + One + Zero + Signed,
{
    type Item = Point<T>;
    fn next(&mut self) -> Option<Self::Item> {
        let two = || T::one() + T::one();

        let to_return = if self.u_is_x {
            (self.u.clone(), self.v.clone())
        } else {
            // flip values if needed
            (self.v.clone(), self.u.clone())
        }
        .into();

        if self.u < self.u_end {
            // always move in u-direction
            self.u = self.u.clone() + T::one();

            self.err = self.err.clone() + self.dv.clone();

            if self.err.clone() * two() >= self.du {
                self.err = self.err.clone() - self.du.clone();
                // now move one in v-direction
                self.v = self.v.clone() + self.v_increment.clone();
            }

            Some(to_return)
        } else if self.u == self.u_end {
            // the last point
            self.u = self.u.clone() + T::one();
            Some(to_return)
        } else {
            // done
            None
        }
    }
}

impl<T: fmt::Display> fmt::Display for Line<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}, {}, {}, {}", self.x1, self.y1, self.x2, self.y2)
    }
}

impl<T> From<[T; 4]> for Line<T> {
    fn from([x1, y1, x2, y2]: [T; 4]) -> Self {
        Self { x1, y1, x2, y2 }
    }
}

impl<T> From<[[T; 2]; 2]> for Line<T> {
    fn from([[x1, y1], [x2, y2]]: [[T; 2]; 2]) -> Self {
        Self { x1, y1, x2, y2 }
    }
}

impl<T> From<[Point<T>; 2]> for Line<T> {
    fn from([begin, end]: [Point<T>; 2]) -> Self {
        let [x1, y1]: [T; 2] = begin.into();
        let [x2, y2]: [T; 2] = end.into();
        Self { x1, y1, x2, y2 }
    }
}

impl<T> From<(T, T, T, T)> for Line<T> {
    fn from((x1, y1, x2, y2): (T, T, T, T)) -> Self {
        Self { x1, y1, x2, y2 }
    }
}

impl<T> From<((T, T), (T, T))> for Line<T> {
    fn from(((x1, y1), (x2, y2)): ((T, T), (T, T))) -> Self {
        Self { x1, y1, x2, y2 }
    }
}

impl<T> From<(Point<T>, Point<T>)> for Line<T> {
    fn from((begin, end): (Point<T>, Point<T>)) -> Self {
        [begin, end].into()
    }
}

impl<T> From<Line<T>> for [T; 4] {
    fn from(Line { x1, y1, x2, y2 }: Line<T>) -> Self {
        [x1, y1, x2, y2]
    }
}

impl<T> From<Line<T>> for [[T; 2]; 2] {
    fn from(Line { x1, y1, x2, y2 }: Line<T>) -> Self {
        [[x1, y1], [x2, y2]]
    }
}

impl<T> From<Line<T>> for [Point<T>; 2] {
    fn from(Line { x1, y1, x2, y2 }: Line<T>) -> Self {
        [Point::new(x1, y1), Point::new(x2, y2)]
    }
}

impl<T> From<Line<T>> for (T, T, T, T) {
    fn from(Line { x1, y1, x2, y2 }: Line<T>) -> Self {
        (x1, y1, x2, y2)
    }
}

impl<T> From<Line<T>> for ((T, T), (T, T)) {
    fn from(Line { x1, y1, x2, y2 }: Line<T>) -> Self {
        ((x1, y1), (x2, y2))
    }
}

impl<T> From<Line<T>> for (Point<T>, Point<T>) {
    fn from(Line { x1, y1, x2, y2 }: Line<T>) -> Self {
        (Point::new(x1, y1), Point::new(x2, y2))
    }
}
