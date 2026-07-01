#pragma once
#include <oven/tray/tray.hpp>

#define TRAY_DECLARE_BINOP(binop) \
Tray binop(const Tray&, const Tray&);\
Tray binop(const Tray&, const Scalar&);\
Tray binop(const Scalar&, const Tray&);\

namespace oven {
TRAY_DECLARE_BINOP(add);
TRAY_DECLARE_BINOP(sub);
TRAY_DECLARE_BINOP(mul);
TRAY_DECLARE_BINOP(div);
TRAY_DECLARE_BINOP(le);
TRAY_DECLARE_BINOP(leq);
TRAY_DECLARE_BINOP(ge);
TRAY_DECLARE_BINOP(geq);
TRAY_DECLARE_BINOP(eq);
TRAY_DECLARE_BINOP(neq);

Tray where(const Tray& predicate, const Tray& self, const Tray& other);

Tray gather(const Tray& self, int64_t dim, const Tray& index);
void scatter_(const Tray& self, int64_t dim, const Tray& index, const Tray& src);

Tray exp(const Tray& self);
Tray neg(const Tray& self);
Tray recip(const Tray& self);
Tray log(const Tray& self);
}