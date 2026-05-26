use iced_widget::{Column, Toggler, button, column, row, text};

use crate::ui::UserInput;

#[derive(Debug, Clone, PartialEq)]
pub struct Cut {
    pub x: bool,
    pub x_bound: f32,
    pub x_inverse: bool,
    x_inv: f32,
    pub y: bool,
    pub y_bound: f32,
    pub y_inverse: bool,
    y_inv: f32,
    // pub z: bool,
    // pub z_bound: f32,
}

impl Default for Cut {
    fn default() -> Self {
        Self {
            x: false,
            x_bound: 0.,
            x_inverse: false,
            x_inv: 1.,
            y: false,
            y_bound: 0.,
            y_inverse: false,
            y_inv: 1.,
        }
    }
}

impl Cut {
    pub fn cut(&self, position: &[f64; 3]) -> bool {
        if self.x && self.y {
            self.x_inv * (position[0] as f32 - self.x_bound) >= 0.
                && self.y_inv * (position[1] as f32 - self.y_bound) >= 0.
        } else if self.x {
            self.x_inv * (position[0] as f32 - self.x_bound) >= 0.
        } else if self.y {
            self.y_inv * (position[1] as f32 - self.y_bound) >= 0.
        } else {
            true
        }
    }
    pub fn x_flip(&mut self) {
        self.x_inverse = !self.x_inverse;
        self.x_inv *= -1.;
    }
    pub fn y_flip(&mut self) {
        self.y_inverse = !self.y_inverse;
        self.y_inv *= -1.;
    }
    pub fn view(&self) -> Column<'_, UserInput> {
        let x_condition = if self.x_inverse {
            "<=".to_string()
        } else {
            ">=".to_string()
        };
        let cut_x: row::Row<'_, UserInput> = row![
            Toggler::new(self.x)
                .label("Show half-space for:")
                .on_toggle(|_| UserInput::ToggleCutX),
            text(format!(" x {x_condition} ")),
            text(self.x_bound),
            text(" "),
            button("I")
                .on_press(UserInput::FlipCutX)
                .width(28)
                .height(28),
            button("+")
                .on_press(UserInput::CutXBoundChanged(1.))
                .width(28)
                .height(28),
            button("-")
                .on_press(UserInput::CutXBoundChanged(-1.))
                .width(28)
                .height(28),
        ]
        .width(500);

        let y_condition = if self.y_inverse {
            "<=".to_string()
        } else {
            ">=".to_string()
        };
        let cut_y: row::Row<'_, UserInput> = row![
            Toggler::new(self.y)
                .label("Show half-space for:")
                .on_toggle(|_| UserInput::ToggleCutY),
            text(format!(" y {y_condition} ")),
            text(self.y_bound),
            text(" "),
            button("I")
                .on_press(UserInput::FlipCutY)
                .width(28)
                .height(28),
            button("+")
                .on_press(UserInput::CutYBoundChanged(1.))
                .width(28)
                .height(28),
            button("-")
                .on_press(UserInput::CutYBoundChanged(-1.))
                .width(28)
                .height(28),
        ]
        .width(500);

        column![cut_x, cut_y,]
    }
}
