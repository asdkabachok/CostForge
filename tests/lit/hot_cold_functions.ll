; RUN: %opt %costforge -passes='costforge-ipa' -costforge-verbose -S < %s 2>&1 | %FileCheck %s
;
; Test: Module pass should mark hot_path as hot and error_handler as cold.
;
; CHECK: .text.hot
; CHECK: .text.unlikely

define void @main() {
entry:
  call void @hot_path()
  call void @hot_path()
  call void @hot_path()
  call void @error_handler()
  ret void
}

define void @hot_path() {
entry:
  ; Small hot function — should be inlined or marked hot
  %x = add i32 1, 2
  %y = add i32 %x, 3
  ret void
}

define void @error_handler() {
entry:
  ; Large cold function — should be marked cold
  %a0 = add i32 1, 1
  %a1 = add i32 %a0, 1
  %a2 = add i32 %a1, 1
  %a3 = add i32 %a2, 1
  %a4 = add i32 %a3, 1
  %a5 = add i32 %a4, 1
  %a6 = add i32 %a5, 1
  %a7 = add i32 %a6, 1
  %a8 = add i32 %a7, 1
  %a9 = add i32 %a8, 1
  %b0 = add i32 %a9, 1
  %b1 = add i32 %b0, 1
  %b2 = add i32 %b1, 1
  %b3 = add i32 %b2, 1
  %b4 = add i32 %b3, 1
  %b5 = add i32 %b4, 1
  %b6 = add i32 %b5, 1
  %b7 = add i32 %b6, 1
  %b8 = add i32 %b7, 1
  %b9 = add i32 %b8, 1
  %c0 = add i32 %b9, 1
  %c1 = add i32 %c0, 1
  %c2 = add i32 %c1, 1
  %c3 = add i32 %c2, 1
  %c4 = add i32 %c3, 1
  %c5 = add i32 %c4, 1
  %c6 = add i32 %c5, 1
  %c7 = add i32 %c6, 1
  %c8 = add i32 %c7, 1
  %c9 = add i32 %c8, 1
  call void @abort()
  unreachable
}

declare void @abort() noreturn
