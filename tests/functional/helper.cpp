/**
* @file helper.cpp
 * @brief Implementation of utility functions for the Involute solver.
 */

#include "involute/core/result.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <map>
#include <variant>

namespace involute::utils {

    // Using a variant to allow strings, ints, and doubles in the metadata map
    using MetaValue = std::variant<std::string, int, unsigned int, double, float>;

    void export_meta(const std::map<std::string, MetaValue>& metadata,
                     const std::string& directory_path,
                     const std::string& file_name) {

        namespace fs = std::filesystem;

        if (!directory_path.empty() && !fs::exists(directory_path)) {
            fs::create_directories(directory_path);
        }

        fs::path full_path = fs::path(directory_path) / file_name;
        std::ofstream file(full_path);

        if (!file.is_open()) {
            std::cerr << "[Involute] Error: Could not open file " << full_path << " for writing.\n";
            return;
        }

        file << "{\n";
        for (auto it = metadata.begin(); it != metadata.end(); ++it) {
            file << "  \"" << it->first << "\": ";

            // Visit the variant to determine how to print the value
            std::visit([&file](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, std::string>) {
                    file << "\"" << arg << "\"";
                } else {
                    file << arg;
                }
            }, it->second);

            if (std::next(it) != metadata.end()) {
                file << ",";
            }
            file << "\n";
        }
        file << "}\n";

        file.close();
        std::cout << "[Involute] Successfully exported metadata to: " << full_path << "\n";
    }

    void export_history_to_csv(const std::vector<involute::core::StepRecord>& history,
                               const std::string& directory_path,
                               const std::string& file_name) {

        // Namespace alias for cleaner code
        namespace fs = std::filesystem;

        // 1. Ensure the target directory exists; create it if it doesn't
        if (!directory_path.empty() && !fs::exists(directory_path)) {
            try {
                fs::create_directories(directory_path);
            } catch (const fs::filesystem_error& e) {
                std::cerr << "[Involute] Directory creation error: " << e.what() << "\n";
                return;
            }
        }

        // 2. Construct the safe, cross-platform file path
        fs::path full_path = fs::path(directory_path) / file_name;

        // 3. Open the file and write the data
        std::ofstream file(full_path);
        if (!file.is_open()) {
            std::cerr << "[Involute] Error: Could not open file " << full_path << " for writing.\n";
            return;
        }

        // Write the CSV header
        file << "step,energy,beta,lambda,delta\n";

        // Write the iteration records
        for (const auto& record : history) {
            file << record.step << ","
                 << record.energy << ","
                 << record.beta << ","
                 << record.lambda << ","
                 << record.delta << "\n";
        }

        file.close();
        std::cout << "[Involute] Successfully exported optimization history to: " << full_path << "\n";
    }

} // namespace involute::utils