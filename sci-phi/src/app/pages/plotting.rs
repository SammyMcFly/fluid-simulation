use cosmic::{Renderer, Theme, iced::core::Element};

use crate::app::Message;

#[derive(Debug, Clone, Default)]
pub struct PlottingSettings {}

impl<'a> Into<Element<'a, Message, Theme, Renderer>> for &PlottingSettings {
    fn into(self) -> Element<'a, Message, Theme, Renderer> {
        cosmic::widget::column(vec![]).into()
    }
}

#[derive(Debug, Clone, Default)]
pub struct PlottingViewport {}
