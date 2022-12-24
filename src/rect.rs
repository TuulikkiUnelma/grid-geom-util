use crate::{max, min, Line, Point};

use num::{traits::AsPrimitive, Num};
use serde::{Deserialize, Serialize};

use std::fmt;
use std::iter::{self, FusedIterator};

/// A generic axis-aligned rectangle from the point `(x1, y1)` to `(x2, y2)`.
///
/// The range of values is **inclusive**, meaning that the point `(x2, y2)` is inside the rectangle's area.
/// This also means that the smallest possible area for a rectangle is 1,
/// and a rectangle with the area of 0 is impossible to create.
///
/// If `x1` isn't greater than or equal to `x2`, or if `y1` isn't greater than or equal to `y2`,
/// then using this rectangle's methods is undefined behavior and may panic.
///
/// It's marked as `#[non_exhaustive]` to make people less likely to create such malformed rectangles.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(
    bound(
        serialize = "T: Clone + Serialize",
        deserialize = "T: PartialOrd + Deserialize<'de>"
    ),
    into = "[T; 4]",
    from = "[T; 4]"
)]
#[non_exhaustive]
pub struct Rect<T> {
    pub x1: T,
    pub y1: T,
    pub x2: T,
    pub y2: T,
}

impl<T> Rect<T> {
    /// Creates a new rectangle.
    ///
    /// If x2 or y2 are greater than x1 or y1, it's flipped so that the resulting rectangle will be valid.
    pub fn new(x1: T, y1: T, x2: T, y2: T) -> Self
    where
        T: PartialOrd,
    {
        let [x1, x2] = if x1 <= x2 { [x1, x2] } else { [x2, x1] };
        let [y1, y2] = if y1 <= y2 { [y1, y2] } else { [y2, y1] };
        Self { x1, y1, x2, y2 }
    }

    /// Creates a new rectangle without checking the validity of inputs.
    ///
    /// # Safety
    ///
    /// Make sure that x2 and y2 are greater or equal to x1 and y2 respectively.
    /// Otherwise the returned rectangle will be invalid and likely to cause panics or undefined behavior when used.
    pub unsafe fn new_unchecked(x1: T, y1: T, x2: T, y2: T) -> Self {
        Self { x1, y1, x2, y2 }
    }

    /// Creates a rectangle between the given opposite corner points.
    pub fn from_corners<P>(a: P, b: P) -> Self
    where
        P: Into<Point<T>>,
        T: PartialOrd,
    {
        let [[x1, y1], [x2, y2]]: [[T; 2]; 2] = [a.into().into(), b.into().into()];
        Self { x1, y1, x2, y2 }
    }

    /// Creates a rectangle using the beginning and endpoints of a line as opposite corner points.
    pub fn from_line(l: Line<T>) -> Self
    where
        T: PartialOrd,
    {
        Self::from_corners((l.x1, l.y1), (l.x2, l.y2))
    }

    /// Converts self into a tuple.
    ///
    /// This is for when the type can't be inferred when using just [`Into::into`].
    pub fn into_tuple(self) -> (T, T, T, T) {
        self.into()
    }

    /// Converts self into an array.
    ///
    /// This is for when the type can't be inferred when using just [`Into::into`].
    pub fn into_array(self) -> [T; 4] {
        self.into()
    }

    /// Maps a function to all of the coordinates and returns them as a new rectangle.
    ///
    /// Will rearrange the coordinates to make it a valid rectangle if their order is changed by `f`.
    #[must_use]
    pub fn map<F, U>(self, f: F) -> Rect<U>
    where
        U: PartialOrd,
        F: Fn(T) -> U,
    {
        Rect::new(f(self.x1), f(self.y1), f(self.x2), f(self.y2))
    }

    /// Maps a function to both of the x coordinates and returns it as a new rectangle.
    ///
    /// Will rearrange the coordinates to make it a valid rectangle if their order is changed by `f`.
    #[must_use]
    pub fn map_x<F>(self, f: F) -> Self
    where
        T: PartialOrd,
        F: Fn(T) -> T,
    {
        let nx1 = f(self.x1);
        let nx2 = f(self.x2);
        let (x1, x2) = if nx1 <= nx2 { (nx1, nx2) } else { (nx2, nx1) };

        Rect {
            x1,
            y1: self.y1,
            x2,
            y2: self.y2,
        }
    }

    /// Maps a function to both of the y coordinates and returns it as a new rectangle.
    ///
    /// Will rearrange the coordinates to make it a valid rectangle if their order is changed by `f`.
    #[must_use]
    pub fn map_y<F>(self, f: F) -> Self
    where
        T: PartialOrd,
        F: Fn(T) -> T,
    {
        let ny1 = f(self.y1);
        let ny2 = f(self.y2);
        let (y1, y2) = if ny1 <= ny2 { (ny1, ny2) } else { (ny2, ny1) };

        Rect {
            x1: self.x1,
            y1,
            x2: self.x2,
            y2,
        }
    }

    /// Maps a function to the rectangle's corners and returns it as a new rectangle.
    ///
    /// Will rearrange the coordinates to make it a valid rectangle if their order is changed by `f`.
    #[must_use]
    pub fn map_corners<F, U>(self, f: F) -> Rect<U>
    where
        F: Fn(Point<T>) -> Point<U>,
        U: PartialOrd,
    {
        let (a, b) = self.into();
        Rect::from_corners(f(a), f(b))
    }
}

impl<T: Clone + Num + PartialOrd> Rect<T> {
    /// Creates a new rectangle from the given minimum point and dimensions.
    ///
    /// Same as `rect.from_diff(min_point, [width - 1, height - 1])`.
    pub fn from_dim<P: Into<Point<T>>>(min_point: P, [width, height]: [T; 2]) -> Self {
        Self::from_diff(min_point, [width - T::one(), height - T::one()])
    }

    /// Creates a new rectangle from the given minimum point and differences.
    ///
    /// Same as `rect.from_dim(min_point, [dx + 1, dy + 1])`.
    pub fn from_diff<P: Into<Point<T>>>(min_point: P, [dx, dy]: [T; 2]) -> Self {
        let Point { x, y } = min_point.into();
        Self::new(x.clone(), y.clone(), x + dx, y + dy)
    }

    /// Creates a rectangle around the given centerpoint by using the given x and y extents / half-dimensions.
    ///
    /// The extents should be positive.
    pub fn from_center<P: Into<Point<T>>>(center: P, extents: [T; 2]) -> Self {
        let Point { x, y } = center.into();
        let [xe, ye] = extents;
        Self::new(
            x.clone() - xe.clone(),
            y.clone() - ye.clone(),
            x + xe,
            y + ye,
        )
    }

    /// Returns the width of the rectangle, which is `x2 - x1 + 1`.
    pub fn width(&self) -> T {
        self.x2.clone() - self.x1.clone() + T::one()
    }

    /// Returns the height of the rectangle, which is `y2 - y1 + 1`.
    pub fn height(&self) -> T {
        self.y2.clone() - self.y1.clone() + T::one()
    }

    /// Returns the difference of the x-coordinate of the rectangle, which is `x2 - x1`.
    ///
    /// Same as `width - 1`.
    pub fn diff_x(&self) -> T {
        self.x2.clone() - self.x1.clone()
    }

    /// Returns the difference of the y-coordinate of the rectangle, which is `y2 - y1`.
    ///
    /// Same as `height - 1`.
    pub fn diff_y(&self) -> T {
        self.y2.clone() - self.y1.clone()
    }

    /// Returns `[width, height]`.
    pub fn dim(&self) -> [T; 2] {
        [self.width(), self.height()]
    }

    /// Returns `[diff_x, diff_y]`.
    pub fn diff_dim(&self) -> [T; 2] {
        [self.diff_x(), self.diff_y()]
    }

    /// Returns the pair of starting point and dimensions: `[[x1, y1], [width, height]]`.
    pub fn start_dim(&self) -> [[T; 2]; 2] {
        [self.min_point().into(), self.dim()]
    }

    /// Returns `width + height`.
    pub fn dim_sum(&self) -> T {
        self.width() + self.height()
    }

    /// Returns the greater of width and height.
    pub fn max_dim(&self) -> T {
        max(self.width(), self.height())
    }

    /// Returns the smaller of width and height.
    pub fn min_dim(&self) -> T {
        min(self.width(), self.height())
    }

    /// Returns the greater of diff_x and diff_y.
    pub fn max_diff(&self) -> T {
        max(self.diff_x(), self.diff_y())
    }

    /// Returns the smaller of diff_x and diff_y.
    pub fn min_diff(&self) -> T {
        min(self.diff_x(), self.diff_y())
    }

    /// Returns the area of the rectangle, which is `width * height`.
    pub fn area(&self) -> T {
        self.width() * self.height()
    }

    /// Returns the product of the differences rectangle's x and y coordinates.
    ///
    /// Same as `diff_x * diff_y` or `(width - 1) * (height - 1)`.
    pub fn diff_prod(&self) -> T {
        self.diff_x() * self.diff_y()
    }

    /// Returns how square this rectangle is from `[0, 1]`.
    ///
    /// Returns the ratio `min_dim / max_dim`
    pub fn squareness(&self) -> f32
    where
        T: AsPrimitive<f32>,
    {
        self.min_dim().as_() / self.max_dim().as_()
    }

    /// Returns the midpoint of the rectangle.
    pub fn mid(&self) -> Point<T> {
        let two = || T::one() + T::one();
        let two = || Point::new(two(), two());
        (self.min_point() + self.max_point()) / two()
    }

    /// Shrinks this rectangle by the given x and y margins.
    ///
    /// If width or height of the rectangle would shrink under 1, it's capped around the midpoint.
    pub fn shrink(&self, [margin_x, margin_y]: [T; 2]) -> Rect<T> {
        let two = || T::one() + T::one();
        let Rect { x1, y1, x2, y2 } = self.clone();

        let (x1, x2) = if margin_x.clone() * two() < self.diff_x() {
            (x1 + margin_x.clone(), x2 - margin_x)
        } else {
            let m = (x2 + x1) / two();
            (m.clone(), m)
        };

        let (y1, y2) = if margin_y.clone() * two() < self.diff_y() {
            (y1 + margin_y.clone(), y2 - margin_y)
        } else {
            let m = (y2 + y1) / two();
            (m.clone(), m)
        };

        Rect { x1, y1, x2, y2 }
    }

    /// Grows this rectangle by the given x and y margins.
    ///
    /// Similar to [`Rect::shrink`] except that this doesn't do the width and height checks,
    /// and will happily shrink the rectangle under the width or height of 1 if the margins are negative and too big.
    /// The returned rectangle will however be valid as the coordinates are flipped by [`Rect::new`] if needed.
    pub fn grow(&self, [margin_x, margin_y]: [T; 2]) -> Rect<T> {
        let Rect { x1, y1, x2, y2 } = self.clone();
        Rect::new(
            x1 - margin_x.clone(),
            y1 - margin_y.clone(),
            x2 + margin_x,
            y2 + margin_y,
        )
    }

    /// Returns true if this rectangle fits within the given `larger` rectangle.
    ///
    /// Sharing borders (eg. both x1 values are the same) counts as being inside.
    /// This means that using the same rectangle as both arguments will return true.
    pub fn is_inside(&self, larger: &Self) -> bool {
        self.corners_iter().all(|p| p.is_inside(larger))
    }

    /// Returns a new rectangle that's the same shape but moved to fit inside the given `larger` rectangle.
    ///
    /// Returns `None` if this rectangle is wider or taller than the `larger` and therefore would not fit.
    pub fn to_inside(&self, larger: &Self) -> Option<Self> {
        if self.width() > larger.width() || self.height() > larger.height() {
            return None;
        }

        let mut out = self.clone();

        // move along x if needed
        if self.x1 < larger.x1 {
            out = out.map_x(|x| x + (larger.x1.clone() - self.x1.clone()));
        } else if self.x2 > larger.x2 {
            out = out.map_x(|x| x - (self.x2.clone() - larger.x2.clone()));
        }

        // move along y if needed
        if self.y1 < larger.y1 {
            out = out.map_y(|y| y + (larger.y1.clone() - self.y1.clone()));
        } else if self.y2 > larger.y2 {
            out = out.map_y(|y| y - (self.y2.clone() - larger.y2.clone()));
        }

        Some(out)
    }

    /// Returns true if this rectangle intersects the given rectangle.
    pub fn does_intersect(&self, other: &Self) -> bool {
        let [ax1, ay1, ax2, ay2] = self.clone().into_array();
        let [bx1, by1, bx2, by2] = other.clone().into_array();
        ax1 <= bx2 && ax2 >= bx1 && ay1 <= by2 && ay2 >= by1
    }

    /// Intersects this rectangle with another and returns the intersection if it exists.
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        if self.does_intersect(other) {
            let [ax1, ay1, ax2, ay2] = self.clone().into_array();
            let [bx1, by1, bx2, by2] = other.clone().into_array();
            Some(Self::new(
                max(ax1, bx1),
                max(ay1, by1),
                min(ax2, bx2),
                min(ay2, by2),
            ))
        } else {
            None
        }
    }

    /// Splits this rectangle to a group of several sub-rectangles.
    ///
    /// If either of `cols` or `rows` is a non-positive number, this will return an empty vector.
    ///
    /// The returned rectangles will be evenly split into the given columns and rows.
    /// They are in order of increasing coordinates, with rows grouped together.
    ///
    /// If x or y range can't be evenly split into the given amount of columns or rows,
    /// then the last sub-rectangle will be of smaller size.
    ///
    /// If the given amount of columns or rows is greater than the rectangle's width or height respectively,
    /// then only width or height amount of columns or rows are returned.
    ///
    /// The returned rectangles will share borders.
    ///
    /// For example if this rectangle goes from `x1 = 0` to `x2 = 10`, and you split it to 2 columns,
    /// you will get rectangles that go from `x1 = 0` to `x2 = 5` and from `x1 = 5` to `x2 = 10`.
    pub fn split(&self, cols: T, rows: T) -> Vec<Self>
    where
        T: AsPrimitive<usize>,
    {
        if cols <= T::zero() || rows <= T::zero() {
            return Vec::new();
        }

        let mut out = Vec::with_capacity(cols.as_() * rows.as_());

        let delta_x = self.width() / cols;
        let delta_y = self.height() / cols;
        let (mut cur_x, mut cur_y) = self.min_point().into();
        let (mut old_x, mut old_y) = (cur_x, cur_y);

        for _ in 0..rows.as_() {
            cur_y = min(cur_y + delta_y, self.y2);
            for _ in 0..cols.as_() {
                cur_x = min(cur_x + delta_x, self.x2);
                out.push(Rect::new(old_x, old_y, cur_x, cur_y));
                old_x = cur_x
            }
            old_y = cur_y;
        }

        out
    }

    /// Splits this rectangle to 4 parts, using the given splitting point.
    ///
    /// Returns `None` if the splitting point is outside this rectangle.
    ///
    /// They are returned in the increasing order of coordinates, with the top row first.
    ///
    /// The returned rectangles will share borders.
    pub fn split_quad(&self, splitting_point: Point<T>) -> Option<[Self; 4]> {
        if !splitting_point.is_inside(self) {
            return None;
        }

        let Rect { x1, y1, x2, y2 } = self.clone();
        let (px, py) = splitting_point.into();
        Some(unsafe {
            [
                Rect::new_unchecked(x1.clone(), y1.clone(), px.clone(), py.clone()),
                Rect::new_unchecked(px.clone(), y1, x2.clone(), py.clone()),
                Rect::new_unchecked(x1, py.clone(), px.clone(), y2.clone()),
                Rect::new_unchecked(px, py, x2, y2),
            ]
        })
    }

    /// Horizontally splits this rectangle to 2 parts, using the given splitting line.
    ///
    /// Returns `None` if the splitting line is outside this rectangle.
    ///
    /// The one with smaller coordinates is returned first.
    ///
    /// The returned rectangles will share borders.
    pub fn split_x(&self, split_x: T) -> Option<[Self; 2]> {
        let Rect { x1, y1, x2, y2 } = self.clone();
        let px = split_x;

        if px < x1 || x2 < px {
            return None;
        }

        Some(unsafe {
            [
                Rect::new_unchecked(x1, y1.clone(), px.clone(), y2.clone()),
                Rect::new_unchecked(px, y1, x2, y2),
            ]
        })
    }

    /// Vertically splits this rectangle to 2 parts, using the given splitting line.
    ///
    /// Returns `None` if the splitting line is outside this rectangle.
    ///
    /// The one with smaller coordinates is returned first.
    ///
    /// The returned rectangles will share borders.
    pub fn split_y(&self, split_y: T) -> Option<[Self; 2]> {
        let Rect { x1, y1, x2, y2 } = self.clone();
        let py = split_y;

        if py < y1 || y2 < py {
            return None;
        }

        Some(unsafe {
            [
                Rect::new_unchecked(x1.clone(), y1, x2.clone(), py.clone()),
                Rect::new_unchecked(x1, py, x2, y2),
            ]
        })
    }

    /// Returns an iterator over the columns of the rectangle (`x1..=x2`).
    ///
    /// The values are always separated by the distance of 1.
    pub fn cols_iter(&self) -> impl Clone + Iterator<Item = T> + FusedIterator {
        let Rect { x1, x2, .. } = self.clone();
        iter::successors(Some(x1), move |x| {
            if *x < x2 {
                Some(x.clone() + T::one())
            } else {
                None
            }
        })
    }

    /// Returns an iterator over the rows of the rectangle (`y1..=y2`).
    ///
    /// The values are always separated by the distance of 1.
    pub fn rows_iter(&self) -> impl Clone + Iterator<Item = T> + FusedIterator {
        let Rect { y1, y2, .. } = self.clone();
        iter::successors(Some(y1), move |y| {
            if *y < y2 {
                Some(y.clone() + T::one())
            } else {
                None
            }
        })
    }

    /// Returns an iterator over all of the points of this rectangle.
    ///
    /// The returned order is left-to-right (from x1 to x2), and then up-to-down (from y1 to y2).
    ///
    /// The points are in a grid with the separation of 1 in both axises.
    ///
    /// The borders are included.
    pub fn points_iter(&self) -> impl Clone + Iterator<Item = Point<T>> + FusedIterator {
        let Rect { x1, y1, x2, y2 } = self.clone();
        iter::successors(Some(Point::new(x1.clone(), y1)), move |p| {
            if p.x < x2 {
                Some(p.clone().map_x(|x| x + T::one()))
            } else if p.y < y2 {
                Some(Point::new(x1.clone(), p.y.clone() + T::one()))
            } else {
                None
            }
        })
    }

    /// Returns an iterator over all of the border points of this rectangle.
    ///
    /// The returned order is clockwise first from `(x1, y1)` to `(x2, y1)`, then to `(x2, y2)`,
    /// then to `(x1, y2)`, and then back to `(x1, y1)` (excluding the `(x1, y1)` point).
    /// No duplicate points will be returned.
    ///
    /// The points are in a grid with the separation of 1 in both axises.
    pub fn points_border_iter(
        &self,
    ) -> impl Clone + Iterator<Item = Point<T>> + FusedIterator + ExactSizeIterator + DoubleEndedIterator
    {
        let Rect { x1, y1, x2, y2 } = self.clone();

        // for the edge cases
        let w_is_1 = self.diff_x() == T::zero();
        let h_is_1 = self.diff_y() == T::zero();

        let mut points = Vec::new();
        let mut f = |x: &T, y: &T| points.push(Point::new(x.clone(), y.clone()));

        // (x1..=x2, y1)
        let mut x = x1.clone();
        while x <= x2 {
            f(&x, &y1);
            x = x + T::one();
        }
        if !h_is_1 {
            // (x2, y1+1..=y2)
            let mut y = y1.clone() + T::one();
            while y <= y2 {
                f(&x2, &y);
                y = y + T::one();
            }

            if !w_is_1 {
                // (x2-1..x1, y2)
                let mut x = x2 - T::one();
                while x > x1 {
                    f(&x, &y2);
                    x = x - T::one();
                }

                // (x1, y2..y1)
                let mut y = y2;
                while y > y1 {
                    f(&x1, &y);
                    y = y - T::one();
                }
            }
        }

        points.into_iter()
    }

    /// Returns an iterator over the corners of this rectangle
    pub fn corners_iter(
        &self,
    ) -> impl Clone + Iterator<Item = Point<T>> + FusedIterator + ExactSizeIterator + DoubleEndedIterator
    {
        std::iter::IntoIterator::into_iter([
            self.min_point(),
            self.x1y2(),
            self.max_point(),
            self.x2y1(),
        ])
    }

    /// Returns the point `(x1, y1)`.
    pub fn min_point(&self) -> Point<T> {
        Point::new(self.x1.clone(), self.y1.clone())
    }

    /// Returns the point `(x2, y2)`.
    pub fn max_point(&self) -> Point<T> {
        Point::new(self.x2.clone(), self.y2.clone())
    }

    /// Returns the point `(x1, y2)`.
    pub fn x1y2(&self) -> Point<T> {
        Point::new(self.x1.clone(), self.y2.clone())
    }

    /// Returns the point `(x2, y1)`.
    pub fn x2y1(&self) -> Point<T> {
        Point::new(self.x2.clone(), self.y1.clone())
    }

    /// Returns a line between `(x1,y1)` and `(x2,y2)`
    pub fn rising_diagonal(&self) -> Line<T> {
        let Rect { x1, y1, x2, y2 } = self.clone();
        Line { x1, y1, x2, y2 }
    }

    /// Returns a line between `(x1,y2)` and `(x2,y1)`
    pub fn falling_diagonal(&self) -> Line<T> {
        [self.x1y2(), self.x2y1()].into()
    }

    /// Returns the horizontal side with low y, so `(y1, [x1, x2])`.
    pub fn min_abscissa(&self) -> (T, [T; 2]) {
        let Rect { y1, x1, x2, .. } = self.clone();
        (y1, [x1, x2])
    }

    /// Returns the horizontal side with low y as a line, so `(x1, y1) - (x2, y1)`.
    pub fn min_abscissa_line(&self) -> Line<T> {
        let Rect { y1, x1, x2, .. } = self.clone();
        ((x1, y1.clone()), (x2, y1)).into()
    }

    /// Returns the horizontal side with high y, so `(y2, [x1, x2])`.
    pub fn max_abscissa(&self) -> (T, [T; 2]) {
        let Rect { y2, x1, x2, .. } = self.clone();
        (y2, [x1, x2])
    }

    /// Returns the horizontal side with high y as a line, so `(x1, y2) - (x2, y2)`.
    pub fn max_abscissa_line(&self) -> Line<T> {
        let Rect { y2, x1, x2, .. } = self.clone();
        ((x1, y2.clone()), (x2, y2)).into()
    }

    /// Returns the vertical side with low x, so `(x1, [y1, y2])`.
    pub fn min_ordinate(&self) -> (T, [T; 2]) {
        let Rect { x1, y1, y2, .. } = self.clone();
        (x1, [y1, y2])
    }

    /// Returns the vertical side with low x as a line, so `(x1, y1) - (x1, y2)`.
    pub fn min_ordinate_line(&self) -> Line<T> {
        let Rect { x1, y1, y2, .. } = self.clone();
        ((x1.clone(), y1), (x1, y2)).into()
    }

    /// Returns the vertical side with high x, so `(x2, [y1, y2])`.
    pub fn max_ordinate(&self) -> (T, [T; 2]) {
        let Rect { x2, y1, y2, .. } = self.clone();
        (x2, [y1, y2])
    }

    /// Returns the vertical side with high x as a line, so `(x2, y1) - (x2, y2)`.
    pub fn max_ordinate_line(&self) -> Line<T> {
        let Rect { x2, y1, y2, .. } = self.clone();
        ((x2.clone(), y1), (x2, y2)).into()
    }
}

impl<T: fmt::Display> fmt::Display for Rect<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}, {}, {}, {}", self.x1, self.y1, self.x2, self.y2)
    }
}

impl<T: PartialOrd> From<[T; 4]> for Rect<T> {
    fn from([x1, y1, x2, y2]: [T; 4]) -> Self {
        Self::new(x1, y1, x2, y2)
    }
}

impl<T: PartialOrd> From<[[T; 2]; 2]> for Rect<T> {
    fn from([[x1, y1], [x2, y2]]: [[T; 2]; 2]) -> Self {
        Self::new(x1, y1, x2, y2)
    }
}

impl<T: PartialOrd> From<[Point<T>; 2]> for Rect<T> {
    /// Same as [`Rect::from_corners`].
    fn from([min, max]: [Point<T>; 2]) -> Self {
        Self::from_corners(min, max)
    }
}

impl<T: PartialOrd> From<(T, T, T, T)> for Rect<T> {
    fn from((x1, y1, x2, y2): (T, T, T, T)) -> Self {
        Self::new(x1, y1, x2, y2)
    }
}

impl<T: PartialOrd> From<((T, T), (T, T))> for Rect<T> {
    fn from(((x1, y1), (x2, y2)): ((T, T), (T, T))) -> Self {
        Self::new(x1, y1, x2, y2)
    }
}

impl<T: PartialOrd> From<(Point<T>, Point<T>)> for Rect<T> {
    fn from((min, max): (Point<T>, Point<T>)) -> Self {
        Self::from_corners(min, max)
    }
}

impl<T> From<Rect<T>> for [T; 4] {
    fn from(Rect { x1, y1, x2, y2 }: Rect<T>) -> Self {
        [x1, y1, x2, y2]
    }
}

impl<T> From<Rect<T>> for [[T; 2]; 2] {
    fn from(Rect { x1, y1, x2, y2 }: Rect<T>) -> Self {
        [[x1, y1], [x2, y2]]
    }
}

impl<T> From<Rect<T>> for [Point<T>; 2] {
    fn from(Rect { x1, y1, x2, y2 }: Rect<T>) -> Self {
        [Point::new(x1, y1), Point::new(x2, y2)]
    }
}

impl<T> From<Rect<T>> for (T, T, T, T) {
    fn from(Rect { x1, y1, x2, y2 }: Rect<T>) -> Self {
        (x1, y1, x2, y2)
    }
}

impl<T> From<Rect<T>> for ((T, T), (T, T)) {
    fn from(Rect { x1, y1, x2, y2 }: Rect<T>) -> Self {
        ((x1, y1), (x2, y2))
    }
}

impl<T> From<Rect<T>> for (Point<T>, Point<T>) {
    fn from(Rect { x1, y1, x2, y2 }: Rect<T>) -> Self {
        (Point::new(x1, y1), Point::new(x2, y2))
    }
}
