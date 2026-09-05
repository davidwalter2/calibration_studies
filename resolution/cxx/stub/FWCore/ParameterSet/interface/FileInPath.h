#ifndef FWCore_ParameterSet_FileInPath_h
#define FWCore_ParameterSet_FileInPath_h
// STANDALONE STUB, for the ctypes shim only.
//
// `cvhcf::CvhCfExponents` resolves its Moliere shape table through
// edm::FileInPath, which needs the CMSSW runtime. Outside it, the release
// source tree is named by $CMSSW_SRC -- the same variable build.sh already
// uses to find the sources it compiles -- so the shim reads the SAME table
// file the maker will.
#include <cstdlib>
#include <string>
namespace edm {
  class FileInPath {
  public:
    explicit FileInPath(const std::string &rel) : rel_(rel) {}
    std::string fullPath() const {
      const char *b = std::getenv("CMSSW_SRC");
      return std::string(b ? b : ".") + "/" + rel_;
    }

  private:
    std::string rel_;
  };
}  // namespace edm
#endif
