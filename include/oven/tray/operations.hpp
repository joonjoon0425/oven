#pragma once
#include <oven/tray/tray.hpp>

namespace oven {
Tray add(const Tray& self, const Tray& other);
Tray sub(const Tray& self, const Tray& other);
Tray mul(const Tray& self, const Tray& other);
Tray div(const Tray& self, const Tray& other);

Tray leq(const Tray& self, const Tray& other);
Tray geq(const Tray& self, const Tray& other);
Tray le(const Tray& self, const Tray& other);
Tray ge(const Tray& self, const Tray& other);
Tray eq(const Tray& self, const Tray& other);
Tray neq(const Tray& self, const Tray& other);

Tray where(const Tray& predicate, const Tray& self, const Tray& other);

Tray gather(const Tray& self, int64_t dim, const Tray& index);
void scatter_(const Tray& self, int64_t dim, const Tray& index, const Tray& src);

Tray exp(Tray& self);
}