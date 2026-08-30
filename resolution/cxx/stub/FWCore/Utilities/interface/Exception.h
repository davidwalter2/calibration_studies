// Minimal stand-in so CGFQoPBlock.cc compiles OUTSIDE CMSSW.
//
// The offline CF calls only the pure math (blockExponent and below), which
// never throws. cms::Exception appears solely in `configure`, the
// ParameterSet reader, which the shim does not call -- but the file must still
// compile, so the type has to exist and behave like a streamable exception.
#ifndef CVH_STUB_FWCORE_EXCEPTION_H
#define CVH_STUB_FWCORE_EXCEPTION_H
#include <sstream>
#include <stdexcept>
#include <string>

namespace cms {
  class Exception : public std::runtime_error {
  public:
    explicit Exception(const std::string &category)
        : std::runtime_error(category), cat_(category) {}
    template <typename T>
    Exception &operator<<(const T &v) {
      std::ostringstream o;
      o << v;
      msg_ += o.str();
      return *this;
    }
    const char *what() const noexcept override {
      full_ = cat_ + ": " + msg_;
      return full_.c_str();
    }

  private:
    std::string cat_, msg_;
    mutable std::string full_;
  };
}  // namespace cms
#endif
