// Minimal stand-in: see the Exception stub. `configure` is compiled but never
// called by the shim; every getParameter here would throw if it were.
#ifndef CVH_STUB_FWCORE_PARAMETERSET_H
#define CVH_STUB_FWCORE_PARAMETERSET_H
#include <stdexcept>
#include <string>

namespace edm {
  class ParameterSet {
  public:
    template <typename T>
    T getParameter(const std::string &) const {
      throw std::logic_error(
          "stub ParameterSet: the offline CF shim must not call cvhcgf::configure");
    }
  };
}  // namespace edm
#endif
