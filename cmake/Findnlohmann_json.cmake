find_package(nlohmann_json CONFIG QUIET)
if(NOT nlohmann_json_FOUND)
    #message(STATUS "NLOHMANN_JSON_PATH is ${NLOHMANN_JSON_PATH}")
    find_path(NLOHMANN_INCLUDE_DIR nlohmann_json.hpp PATH ${NLOHMANN_JSON_PATH})
    message(STATUS "nlohmann include dir: ${NLOHMANN_INCLUDE_DIR}")
    add_library(nlohmann_json INTERFACE IMPORTED)
    set_target_properties(nlohmann_json PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES ${NLOHMANN_INCLUDE_DIR})
endif()
