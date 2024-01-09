/*

TopkAttention

使用pybind11库的C++代码，用于在Python中绑定两个函数，ms_deform_attn_forward 和 ms_deform_attn_backward
PYBIND11_MODULE 是pybind11的一个宏，用于定义一个Python模块。
TORCH_EXTENSION_NAME是模块的名字，而m是一个代表模块的对象，我们可以在其上定义各种Python功能
*/


#include "topk_attn.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("topk_attn_forward", &topk_attn_forward, "topk_attn_forward");
  m.def("topk_attn_backward", &topk_attn_backward, "topk_attn_backward");
}
