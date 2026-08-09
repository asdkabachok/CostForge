; RUN: %opt %costforge -passes='costforge' -costforge-verbose -S < %s 2>&1 | %FileCheck %s
;
; Test: Chain of 5 icmp eq against constants on the same value
;       should be lowered to a switch instruction.
;
; CHECK: switch

define i32 @dispatch(i32 %cmd) {
entry:
  %c0 = icmp eq i32 %cmd, 0
  br i1 %c0, label %case0, label %check1

check1:
  %c1 = icmp eq i32 %cmd, 1
  br i1 %c1, label %case1, label %check2

check2:
  %c2 = icmp eq i32 %cmd, 2
  br i1 %c2, label %case2, label %check3

check3:
  %c3 = icmp eq i32 %cmd, 3
  br i1 %c3, label %case3, label %check4

check4:
  %c4 = icmp eq i32 %cmd, 4
  br i1 %c4, label %case4, label %default

case0:  ret i32 10
case1:  ret i32 20
case2:  ret i32 30
case3:  ret i32 40
case4:  ret i32 50
default: ret i32 -1
}
