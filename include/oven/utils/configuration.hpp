#pragma once

#include "assert.hpp"
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <string>

namespace oven {

class Configuration {
public:
    void open(const std::string& path) {
        std::ifstream file(path);
        OVEN_ASSERT(file.is_open(), "Failed to open config file: " + path + "\nCurrent directory is: " + std::filesystem::current_path().string());

        file >> json_data_;
    }

    template <typename T>
    T get(const std::string& key) const {
        auto ptr = nlohmann::json::json_pointer(key);
        OVEN_ASSERT(json_data_.contains(ptr), "Config pointer path not found: " + key);
        return json_data_[ptr].get<T>();
    }

    template <typename T>
    T get_or(const std::string& key, const T& fallback_value) const {
        auto ptr = nlohmann::json::json_pointer(key);
        if(!json_data_.contains(ptr)) {
            return fallback_value;
        }
        return json_data_[ptr].get<T>();
    }

private:
    nlohmann::json json_data_;
};

}// namespace oven