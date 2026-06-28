#include <oven/tray/dispatcher.hpp>

namespace oven::detail {

Dispatcher& Dispatcher::get_instance() {
    // global instance
    static Dispatcher dispatcher;
    return dispatcher;
}

}