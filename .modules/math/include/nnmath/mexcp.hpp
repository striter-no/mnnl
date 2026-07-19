#pragma once

#include <exception>
#include <string>

namespace nn::excp {

class VectorIndexOverflow : public std::exception {
   public:
    VectorIndexOverflow(std::string msg) : message(msg) {}

    const char* what() const noexcept override { return message.c_str(); }

   private:
    std::string message;
};

class MatrixIndexOverflow : public std::exception {
   public:
    MatrixIndexOverflow(std::string msg) : message(msg) {}

    const char* what() const noexcept override { return message.c_str(); }

   private:
    std::string message;
};

class NotImplemented : public std::exception {
   public:
    NotImplemented(std::string msg) : message(msg) {}

    const char* what() const noexcept override { return message.c_str(); }

   private:
    std::string message;
};

class NotANumber : public std::exception {
   public:
    NotANumber(std::string msg) : message(msg) {}

    const char* what() const noexcept override { return message.c_str(); }

   private:
    std::string message;
};

}  // namespace nn::excp
