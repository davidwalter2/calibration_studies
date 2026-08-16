// barkas_shim -- an LD_PRELOAD switch for Geant4's projectile-charge
// corrections, because Geant4 does not have one.
//
// WHY THIS EXISTS
// ---------------
// G4EmCorrections::{Barkas,Bloch,Mott}Correction are the entire z-expansion of
// Geant4's stopping power beyond z^2, and there is no UI command, no
// G4EmParameters flag and no physics-list hook that turns any of them off.
// The decisive test for NOTES_HADRONS s6's charge-odd energy loss is to
// deactivate them in the SIMULATION and watch the charge-oddness vanish, so a
// switch had to be made.
//
// It is made at the dynamic linker.  The calls from
// G4EmCorrections::HighOrderCorrections into the three term functions go
// through the PLT -- verified, not assumed:
//
//   objdump -d libG4processes.so @ HighOrderCorrections:
//     callq  <_ZN15G4EmCorrections16BarkasCorrectionE...@plt>
//     callq  <_ZN15G4EmCorrections15BlochCorrectionE...@plt>
//     callq  <_ZN15G4EmCorrections14MottCorrectionE...@plt>
//   objdump -R libG4processes.so:
//     R_X86_64_JUMP_SLOT  _ZN15G4EmCorrections16BarkasCorrectionE...
//
// so an LD_PRELOAD definition of the same mangled symbols is bound first and
// the interposition reaches Geant4's own internal calls.  (`readelf -r`
// WITHOUT -W wraps these symbol names and makes the JUMP_SLOT entries look
// absent; use -W or objdump -R.)
//
// The shim always calls the original through RTLD_NEXT, so:
//   * with no switch set it is a pure TEE -- it changes nothing and reports
//     what Geant4 used, which is how its own liveness is established;
//   * with a switch set it reports the value it suppressed.
// Either way it prints a call census at exit, so "the switch was live" is a
// number in the log rather than a claim.
//
// Environment:
//   CVH_SHIM_BARKAS_OFF=1   BarkasCorrection -> 0
//   CVH_SHIM_MOTT_OFF=1     MottCorrection   -> 0
//   CVH_SHIM_BLOCH_OFF=1    BlochCorrection  -> 0   (the EVEN control)
//   CVH_SHIM_QUIET=1        no banner / no census
//
// Build: see barkas_probe.py `build`.  Nothing in Geant4 or CMSSW is modified.

#define _GNU_SOURCE
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

namespace {

typedef double (*corr_t)(void*, const void*, const void*, double, bool);

struct State {
  bool barkasOff = false, mottOff = false, blochOff = false, quiet = false;
  long nBarkas = 0, nBloch = 0, nMott = 0;
  // running sums of the ORIGINAL values, so the census reports the size of
  // what was (or was not) suppressed
  double sBarkas = 0.0, sBloch = 0.0, sMott = 0.0;
  corr_t oBarkas = nullptr, oBloch = nullptr, oMott = nullptr;
};

State& st() {
  static State s;
  return s;
}

bool envon(const char* k) {
  const char* v = getenv(k);
  return v && *v && strcmp(v, "0") != 0;
}

void census() {
  State& s = st();
  if (s.quiet)
    return;
  fprintf(stderr,
          "[barkasshim] pid %d census: Barkas n=%ld sum=%.6g %s | "
          "Bloch n=%ld sum=%.6g %s | Mott n=%ld sum=%.6g %s\n",
          (int)getpid(), s.nBarkas, s.sBarkas, s.barkasOff ? "SUPPRESSED" : "passed", s.nBloch,
          s.sBloch, s.blochOff ? "SUPPRESSED" : "passed", s.nMott, s.sMott,
          s.mottOff ? "SUPPRESSED" : "passed");
  fflush(stderr);
}

__attribute__((constructor)) void init() {
  State& s = st();
  s.barkasOff = envon("CVH_SHIM_BARKAS_OFF");
  s.mottOff = envon("CVH_SHIM_MOTT_OFF");
  s.blochOff = envon("CVH_SHIM_BLOCH_OFF");
  s.quiet = envon("CVH_SHIM_QUIET");
  atexit(census);
  if (!s.quiet) {
    fprintf(stderr, "[barkasshim] loaded pid %d: barkasOff=%d blochOff=%d mottOff=%d\n",
            (int)getpid(), (int)s.barkasOff, (int)s.blochOff, (int)s.mottOff);
    fflush(stderr);
  }
}

}  // namespace

// The mangled names, spelled out.  They are the ones `nm -D` prints for
// libG4processes.so; a typo here produces a shim that silently does nothing,
// which is why the call census exists.
#define BARKAS_NAME "_ZN15G4EmCorrections16BarkasCorrectionEPK20G4ParticleDefinitionPK10G4Materialdb"
#define BLOCH_NAME "_ZN15G4EmCorrections15BlochCorrectionEPK20G4ParticleDefinitionPK10G4Materialdb"
#define MOTT_NAME "_ZN15G4EmCorrections14MottCorrectionEPK20G4ParticleDefinitionPK10G4Materialdb"

extern "C" {

double _ZN15G4EmCorrections16BarkasCorrectionEPK20G4ParticleDefinitionPK10G4Materialdb(
    void* self, const void* p, const void* m, double e, bool init) {
  State& s = st();
  if (!s.oBarkas)
    s.oBarkas = (corr_t)dlsym(RTLD_NEXT, BARKAS_NAME);
  const double v = s.oBarkas(self, p, m, e, init);
  ++s.nBarkas;
  s.sBarkas += v;
  return s.barkasOff ? 0.0 : v;
}

double _ZN15G4EmCorrections15BlochCorrectionEPK20G4ParticleDefinitionPK10G4Materialdb(
    void* self, const void* p, const void* m, double e, bool init) {
  State& s = st();
  if (!s.oBloch)
    s.oBloch = (corr_t)dlsym(RTLD_NEXT, BLOCH_NAME);
  const double v = s.oBloch(self, p, m, e, init);
  ++s.nBloch;
  s.sBloch += v;
  return s.blochOff ? 0.0 : v;
}

double _ZN15G4EmCorrections14MottCorrectionEPK20G4ParticleDefinitionPK10G4Materialdb(
    void* self, const void* p, const void* m, double e, bool init) {
  State& s = st();
  if (!s.oMott)
    s.oMott = (corr_t)dlsym(RTLD_NEXT, MOTT_NAME);
  const double v = s.oMott(self, p, m, e, init);
  ++s.nMott;
  s.sMott += v;
  return s.mottOff ? 0.0 : v;
}

}  // extern "C"
