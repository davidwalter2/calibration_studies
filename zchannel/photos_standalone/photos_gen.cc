// Standalone Photos++ 3.61 generator of the Z -> mu mu FSR kernel.
//
// Builds q qbar -> Z -> mu+ mu- at a pre-FSR mass drawn from the UL16 DY
// sample's own m_pre distribution inside a chosen band, hands it to Photos++
// with a chosen set of switches, and accumulates the post-FSR mass loss
// u = -ln(m_post/m_pre) in the same binning `fit_gen.build_kernel` uses.
//
// Nothing here is tuned: every Photos switch is either the sample's setting or
// is named on the command line, and the kinematics are Born q qbar -> Z/gamma*
// with the standard couplings.
//
// Photos++ is a singleton with static state and is NOT thread safe, so
// parallelism is by separate processes, one per (band, replica).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "Photos/Photos.h"
#include "Photos/photosC.h"
#include "Photos/PhotosHEPEVTEvent.h"
#include "Photos/PhotosHEPEVTParticle.h"

using namespace Photospp;

// ---------------------------------------------------------------- binning --
static const int64_t N_FINE = 100000;     // u in [0, 2), width U_FINE
static const double  U_FINE = 2e-5;
static const int64_t N_TAIL = 200;        // u in [2, 12), width U_TAIL
static const double  U_TAIL = 0.05;
static const int64_t N_LOG  = 160;        // log-spaced LOG_LO .. LOG_HI
static const double  LOG_LO = 1e-6, LOG_HI = 2.0;
static const double  NORAD_U = 1e-5;      // fit_gen.build_kernel's noise floor

// --------------------------------------------------------------- constants --
static const double M_MU   = 0.1056583745;
static const double MZ     = 91.1876;
static const double GZ     = 2.4952;
static const double S2W    = 0.23153999447822571;   // the sample's POWHEG value

// ------------------------------------------------------------------- rng ----
static std::mt19937_64 g_rng;
static inline double urand() {
  // (0,1) open: Photos divides by and takes logs of this.
  for (;;) {
    double u = std::generate_canonical<double, 53>(g_rng);
    if (u > 0.0 && u < 1.0) return u;
  }
}
static double photos_rand() { return urand(); }

// ------------------------------------------------------------ m_pre input --
struct Bands {
  std::vector<double> lo, hi, sumw;
  std::vector<int64_t> i0, i1;
  std::vector<double> edges, hist;
};

static void read_bands(const char *fn, Bands &b) {
  FILE *f = fopen(fn, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", fn); exit(1); }
  int32_t nb = 0;
  if (fread(&nb, 4, 1, f) != 1) { fprintf(stderr, "short read\n"); exit(1); }
  b.lo.resize(nb); b.hi.resize(nb); b.sumw.resize(nb);
  b.i0.resize(nb); b.i1.resize(nb);
  for (int i = 0; i < nb; ++i) {
    double d[3]; int64_t k[2];
    if (fread(d, 8, 3, f) != 3 || fread(k, 8, 2, f) != 2) { fprintf(stderr, "short read\n"); exit(1); }
    b.lo[i] = d[0]; b.hi[i] = d[1]; b.sumw[i] = d[2]; b.i0[i] = k[0]; b.i1[i] = k[1];
  }
  int64_t ne = 0;
  if (fread(&ne, 8, 1, f) != 1) { fprintf(stderr, "short read\n"); exit(1); }
  b.edges.resize(ne); b.hist.resize(ne - 1);
  if ((int64_t)fread(b.edges.data(), 8, ne, f) != ne) { fprintf(stderr, "short read\n"); exit(1); }
  if ((int64_t)fread(b.hist.data(), 8, ne - 1, f) != ne - 1) { fprintf(stderr, "short read\n"); exit(1); }
  fclose(f);
}

// Per-band CDF over the fine m_pre bins, sampled with a uniform draw inside
// the chosen fine bin.
struct MassSampler {
  std::vector<double> cdf;
  const Bands *b; int band;
  void init(const Bands &bb, int ib) {
    b = &bb; band = ib;
    cdf.assign(bb.i1[ib] - bb.i0[ib], 0.0);
    double s = 0.0;
    for (size_t k = 0; k < cdf.size(); ++k) { s += bb.hist[bb.i0[ib] + k]; cdf[k] = s; }
    if (s <= 0.0) { fprintf(stderr, "band %d has no sample weight\n", ib); exit(1); }
    for (size_t k = 0; k < cdf.size(); ++k) cdf[k] /= s;
  }
  double draw() const {
    double r = urand();
    size_t k = std::lower_bound(cdf.begin(), cdf.end(), r) - cdf.begin();
    if (k >= cdf.size()) k = cdf.size() - 1;
    double e0 = b->edges[b->i0[band] + k], e1 = b->edges[b->i0[band] + k + 1];
    return e0 + (e1 - e0) * urand();
  }
};

// ------------------------------------------------------- angular sampling --
// Born q qbar -> gamma*/Z -> mu+ mu-, integrated over nothing: the standard
// dsigma/dcos = A0 (1 + cos^2) + 2 A1 cos with
//   chi(s) = s / (s - MZ^2 + i MZ GZ) / (4 s2w c2w)
//   A0 = Qq^2 Ql^2 - 2 Qq Ql vq vl Re chi + (vq^2+aq^2)(vl^2+al^2) |chi|^2
//   A1 = -2 Qq Ql aq al Re chi + 4 vq aq vl al |chi|^2
// (v = T3 - 2 Q s2w, a = T3).  Only the SHAPE is used.
static double afb_ratio(double s, int quark /*2 = up, 1 = down*/) {
  const double c2w = 1.0 - S2W;
  double Qq = (quark == 2) ? (2.0 / 3.0) : (-1.0 / 3.0);
  double T3q = (quark == 2) ? 0.5 : -0.5;
  double Ql = -1.0, T3l = -0.5;
  double vq = T3q - 2.0 * Qq * S2W, aq = T3q;
  double vl = T3l - 2.0 * Ql * S2W, al = T3l;
  double den_r = s - MZ * MZ, den_i = MZ * GZ;
  double n = s / (4.0 * S2W * c2w);
  double d2 = den_r * den_r + den_i * den_i;
  double chir = n * den_r / d2, chi2 = n * n / d2;
  double A0 = Qq * Qq * Ql * Ql - 2.0 * Qq * Ql * vq * vl * chir
            + (vq * vq + aq * aq) * (vl * vl + al * al) * chi2;
  double A1 = -2.0 * Qq * Ql * aq * al * chir + 4.0 * vq * aq * vl * al * chi2;
  return (A0 > 0.0) ? (A1 / A0) : 0.0;
}

enum AngMode { ANG_BORN, ANG_FLAT, ANG_ASYM };

static double draw_cos(AngMode mode, double R) {
  if (mode == ANG_FLAT) return 2.0 * urand() - 1.0;
  if (mode == ANG_ASYM) return 2.0 * std::cbrt(urand()) - 1.0;   // ~ (1+c)^2
  double fmax = 2.0 + 2.0 * std::fabs(R);
  for (;;) {
    double c = 2.0 * urand() - 1.0;
    double f = (1.0 + c * c) + 2.0 * R * c;
    if (f > 0.0 && urand() * fmax < f) return c;
  }
}

// -------------------------------------------------------------- boost data --
struct BoostSample {
  std::vector<double> m, pt, y;
  void read(const char *fn) {
    FILE *f = fopen(fn, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", fn); exit(1); }
    int64_t n = 0;
    if (fread(&n, 8, 1, f) != 1) { fprintf(stderr, "short read\n"); exit(1); }
    m.resize(n); pt.resize(n); y.resize(n);
    std::vector<double> buf(3 * n);
    if ((int64_t)fread(buf.data(), 8, 3 * n, f) != 3 * n) { fprintf(stderr, "short read\n"); exit(1); }
    for (int64_t i = 0; i < n; ++i) { m[i] = buf[3*i]; pt[i] = buf[3*i+1]; y[i] = buf[3*i+2]; }
    fclose(f);
  }
};

// ------------------------------------------------------------ accumulator --
struct Acc {
  double n = 0, n_nophot = 0, n_noemit = 0, n_norad = 0, mpre_s1 = 0, over = 0;
  double mom[4] = {0, 0, 0, 0};
  double nphot_s1 = 0, npair = 0, nbad = 0;
  std::vector<double> f0, f1, f2, t0, t1, t2, lg;
  Acc() : f0(N_FINE, 0.0), f1(N_FINE, 0.0), f2(N_FINE, 0.0),
          t0(N_TAIL, 0.0), t1(N_TAIL, 0.0), t2(N_TAIL, 0.0), lg(N_LOG, 0.0) {}
  void fill(double u, double mpre, int nph, int npr) {
    n += 1.0; mpre_s1 += mpre;
    if (nph == 0) n_nophot += 1.0;
    // "Photos emitted nothing": the sample's npre == 0 flag is exactly this,
    // since Photos writes the status-746 pre-emission copies whenever it
    // touched the muons -- by a photon or by a pair.
    if (nph == 0 && npr == 0) n_noemit += 1.0;
    if (u <= NORAD_U) n_norad += 1.0;
    nphot_s1 += nph;
    double x = -std::expm1(-2.0 * u);
    mom[0] += 1.0; mom[1] += u; mom[2] += u * u; mom[3] += x;
    if (u < 2.0) {
      int64_t j = (int64_t)(u / U_FINE);
      if (j < 0) j = 0;
      if (j >= N_FINE) j = N_FINE - 1;
      f0[j] += 1.0; f1[j] += u; f2[j] += u * u;
    } else {
      int64_t j = (int64_t)((u - 2.0) / U_TAIL);
      if (j >= 0 && j < N_TAIL) { t0[j] += 1.0; t1[j] += u; t2[j] += u * u; }
      else over += 1.0;
    }
    if (u >= LOG_LO && u < LOG_HI) {
      double lj = std::log(u / LOG_LO) / std::log(LOG_HI / LOG_LO) * (double)N_LOG;
      int64_t j = (int64_t)lj;
      if (j < 0) j = 0;
      if (j >= N_LOG) j = N_LOG - 1;
      lg[j] += 1.0;
    }
  }
};

// ------------------------------------------------------------------- args --
static std::string arg_str(int argc, char **argv, const char *k, const char *def) {
  size_t L = strlen(k);
  for (int i = 1; i < argc; ++i)
    if (!strncmp(argv[i], k, L) && argv[i][L] == '=') return std::string(argv[i] + L + 1);
  return std::string(def);
}
static double arg_d(int argc, char **argv, const char *k, double d) {
  std::string s = arg_str(argc, argv, k, "");
  return s.empty() ? d : atof(s.c_str());
}
static long arg_i(int argc, char **argv, const char *k, long d) {
  std::string s = arg_str(argc, argv, k, "");
  return s.empty() ? d : atol(s.c_str());
}

// ------------------------------------------------------------------- main --
int main(int argc, char **argv) {
  std::string fbands = arg_str(argc, argv, "--bands", "data/photos/mpre_bands.bin");
  std::string fboost = arg_str(argc, argv, "--boost", "data/photos/boost_sample.bin");
  std::string fout   = arg_str(argc, argv, "--out",   "out.bin");
  long nev    = arg_i(argc, argv, "--n", 100000);
  int  b0     = (int)arg_i(argc, argv, "--band-first", 0);
  int  b1     = (int)arg_i(argc, argv, "--band-last", -1);
  long seed   = arg_i(argc, argv, "--seed", 1);
  int  me     = (int)arg_i(argc, argv, "--me", 0);
  int  pairs  = (int)arg_i(argc, argv, "--pairs", 0);
  int  phot   = (int)arg_i(argc, argv, "--phot", 1);
  int  expo   = (int)arg_i(argc, argv, "--exp", 1);
  int  interf = (int)arg_i(argc, argv, "--interf", -1);
  int  isec   = (int)arg_i(argc, argv, "--doublebrem", -1);
  double fint = arg_d(argc, argv, "--fint", 0.0);
  double xph  = arg_d(argc, argv, "--xphcut", 1e-7);
  double alph = arg_d(argc, argv, "--alpha", 0.0);
  int  useboost = (int)arg_i(argc, argv, "--kinboost", 0);
  std::string sang = arg_str(argc, argv, "--ang", "born");
  std::string squark = arg_str(argc, argv, "--quark", "mix");
  double fracu = arg_d(argc, argv, "--frac-up", 0.6);
  int  verbose = (int)arg_i(argc, argv, "--verbose", 0);

  AngMode ang = sang == "flat" ? ANG_FLAT : (sang == "asym" ? ANG_ASYM : ANG_BORN);
  int qfix = squark == "u" ? 2 : (squark == "d" ? 1 : 0);

  Bands B; read_bands(fbands.c_str(), B);
  int nb = (int)B.lo.size();
  if (b1 < 0 || b1 >= nb) b1 = nb - 1;
  BoostSample BS; if (useboost) BS.read(fboost.c_str());

  g_rng.seed((uint64_t)seed * 6364136223846793005ULL + 1442695040888963407ULL);

  // ---- Photos configuration -------------------------------------------
  // Photos::initialize() re-applies setExponentiation(true) (which resets
  // xphcut to 1e-7, isec/itre to 0 and the kinematic corrections to 5) and
  // unconditionally re-applies maxWtInterference(2.0) when interference is on,
  // so everything it touches has to be set again afterwards.
  //
  // phokey.fint is an over-sampling envelope, not a physics parameter: the
  // crude emission probability is multiplied by it (photosC.cxx:1928) and the
  // accept/reject weight divided by it (photosC.cxx:2328).  Photos aborts when
  // the ratio still exceeds 1, which the Z matrix-element correction and pair
  // emission do at the default 2.0; --fint raises the envelope and the
  // resulting kernel is checked to be independent of it.
  Photos::setMomentumUnit(Photos::GEV);
  Photos::setRandomGenerator(photos_rand);
  Photos::setExponentiation(expo != 0);
  Photos::setMeCorrectionWtForZ(me != 0);
  Photos::setPairEmission(pairs != 0);
  Photos::setPhotonEmission(phot != 0);
  if (alph > 0.0) Photos::setAlphaQED(alph);
  if (interf >= 0) Photos::setInterference(interf != 0);
  Photos::setStopAtCriticalError(false);
  Photos::initialize();
  Photos::setInfraredCutOff(xph);
  if (fint > 0.0) Photos::maxWtInterference(fint);
  if (isec >= 0) Photos::setDoubleBrem(isec != 0);
  if (verbose) {
    Photos::iniInfo();
    fprintf(stderr,
            "CONFIG iexp=%d interf=%d isec=%d itre=%d xphcut=%.6e alpha=%.9e "
            "fint=%g expeps=%g me=%d pairs=%d phot=%d ang=%s quark=%s boost=%d\n",
            phokey.iexp, phokey.interf, phokey.isec, phokey.itre,
            phocop.xphcut, phocop.alpha, phokey.fint, phokey.expeps,
            me, pairs, phot, sang.c_str(), squark.c_str(), useboost);
  }

  std::vector<Acc> acc(b1 - b0 + 1);
  long nfail = 0;

  for (int ib = b0; ib <= b1; ++ib) {
    MassSampler MS; MS.init(B, ib);
    Acc &A = acc[ib - b0];
    for (long iev = 0; iev < nev; ++iev) {
      double m = MS.draw(), ptz = 0.0, yz = 0.0;
      if (useboost) {
        // draw (pt, y) from the sample entry whose mass is closest in the
        // random draw's own band: the triples are used as a joint sample, and
        // u is boost-invariant, so only the numerics are being probed.
        size_t k = (size_t)(urand() * BS.m.size());
        if (k >= BS.m.size()) k = BS.m.size() - 1;
        ptz = BS.pt[k]; yz = BS.y[k];
      }
      if (m <= 2.0 * M_MU) continue;

      int quark = qfix ? qfix : (urand() < fracu ? 2 : 1);
      double R = (ang == ANG_BORN) ? afb_ratio(m * m, quark) : 0.0;
      double c = draw_cos(ang, R), s = std::sqrt(std::max(0.0, 1.0 - c * c));
      double ph = 2.0 * M_PI * urand();
      double E = 0.5 * m, p = std::sqrt(std::max(0.0, E * E - M_MU * M_MU));

      double mum[4] = { p * s * std::cos(ph), p * s * std::sin(ph), p * c, E };
      double mup[4] = { -mum[0], -mum[1], -mum[2], E };
      double qk[4]  = { 0.0, 0.0,  0.5 * m, 0.5 * m };
      double qb[4]  = { 0.0, 0.0, -0.5 * m, 0.5 * m };
      double zz[4]  = { 0.0, 0.0, 0.0, m };

      if (useboost) {
        double mt = std::sqrt(m * m + ptz * ptz);
        double bz[4] = { ptz, 0.0, mt * std::sinh(yz), mt * std::cosh(yz) };
        double phz = 2.0 * M_PI * urand();
        double bx = bz[0] * std::cos(phz), by = bz[0] * std::sin(phz);
        double bE = bz[3], bpz = bz[2];
        double bvx = bx / bE, bvy = by / bE, bvz = bpz / bE;
        double b2 = bvx * bvx + bvy * bvy + bvz * bvz;
        double gam = 1.0 / std::sqrt(std::max(1e-300, 1.0 - b2));
        double *vs[5] = { mum, mup, qk, qb, zz };
        for (int k = 0; k < 5; ++k) {
          double *v = vs[k];
          double bp = bvx * v[0] + bvy * v[1] + bvz * v[2];
          double g2 = (b2 > 0.0) ? (gam - 1.0) / b2 : 0.0;
          double e2 = gam * (v[3] + bp);
          v[0] += g2 * bp * bvx + gam * bvx * v[3];
          v[1] += g2 * bp * bvy + gam * bvy * v[3];
          v[2] += g2 * bp * bvz + gam * bvz * v[3];
          v[3] = e2;
        }
      }

      // q qbar -> Z -> mu+ mu-; the quarks are the Z's mothers, which is what
      // the Z matrix-element correction needs to find the incoming couplings.
      int qid = (quark == 2) ? 2 : 1;
      PhotosHEPEVTEvent evt;
      evt.addParticle(new PhotosHEPEVTParticle( qid, PhotosParticle::HISTORY,
                      qk[0], qk[1], qk[2], qk[3], 0.0, -1, -1, 2, 2));
      evt.addParticle(new PhotosHEPEVTParticle(-qid, PhotosParticle::HISTORY,
                      qb[0], qb[1], qb[2], qb[3], 0.0, -1, -1, 2, 2));
      evt.addParticle(new PhotosHEPEVTParticle(23, PhotosParticle::DECAYED,
                      zz[0], zz[1], zz[2], zz[3], m, 0, 1, 3, 4));
      PhotosHEPEVTParticle *p1 = new PhotosHEPEVTParticle(13, PhotosParticle::STABLE,
                      mum[0], mum[1], mum[2], mum[3], M_MU, 2, -1, -1, -1);
      PhotosHEPEVTParticle *p2 = new PhotosHEPEVTParticle(-13, PhotosParticle::STABLE,
                      mup[0], mup[1], mup[2], mup[3], M_MU, 2, -1, -1, -1);
      evt.addParticle(p1);
      evt.addParticle(p2);

      int n0 = evt.getParticleCount();
      evt.process();
      int n1 = evt.getParticleCount();

      if (p1->getPdgID() != 13 || p2->getPdgID() != -13) { ++nfail; ++A.nbad; continue; }
      double ex = p1->getPx() + p2->getPx(), ey = p1->getPy() + p2->getPy();
      double ez = p1->getPz() + p2->getPz(), ee = p1->getE() + p2->getE();
      double m2 = ee * ee - ex * ex - ey * ey - ez * ez;
      if (!(m2 > 0.0) || !(m2 <= m * m * (1.0 + 1e-9))) {
        // m2 > m^2 can only be round-off at u = 0; anything else is a failure
        if (m2 > 0.0 && m2 < m * m * (1.0 + 1e-6)) m2 = m * m;
        else { ++nfail; ++A.nbad; continue; }
      }
      double mpost = std::sqrt(m2);
      double u = -std::log(std::min(mpost / m, 1.0));

      int nph_add = 0, npair_add = 0;
      for (int k = n0; k < n1; ++k) {
        int id = evt.getParticle(k)->getPdgID();
        if (id == 22) ++nph_add; else ++npair_add;
      }
      A.npair += npair_add;
      A.fill(u, m, nph_add, npair_add);
    }
    if (verbose)
      fprintf(stderr, "band %d [%g,%g): n = %g, <u> = %.6e, P(0 gamma) = %.6f\n",
              ib, B.lo[ib], B.hi[ib], acc[ib - b0].n,
              acc[ib - b0].mom[1] / acc[ib - b0].n,
              acc[ib - b0].n_nophot / acc[ib - b0].n);
  }

  // ---- write ------------------------------------------------------------
  FILE *f = fopen(fout.c_str(), "wb");
  if (!f) { fprintf(stderr, "cannot write %s\n", fout.c_str()); exit(1); }
  char magic[8] = "PHGEN02";
  fwrite(magic, 1, 8, f);
  int64_t hdr[4] = { (int64_t)(b1 - b0 + 1), N_FINE, N_TAIL, N_LOG };
  fwrite(hdr, 8, 4, f);
  double hd[4] = { U_FINE, U_TAIL, LOG_LO, LOG_HI };
  fwrite(hd, 8, 4, f);
  for (int ib = b0; ib <= b1; ++ib) {
    const Acc &A = acc[ib - b0];
    int64_t idx = ib;
    fwrite(&idx, 8, 1, f);
    double d[2] = { B.lo[ib], B.hi[ib] };
    fwrite(d, 8, 2, f);
    double s[9] = { A.n, A.n_nophot, A.n_norad, A.mpre_s1, A.over,
                    A.mom[0], A.mom[1], A.mom[2], A.mom[3] };
    fwrite(s, 8, 9, f);
    double e[4] = { A.nphot_s1, A.npair, A.nbad, A.n_noemit };
    fwrite(e, 8, 4, f);
    fwrite(A.f0.data(), 8, N_FINE, f);
    fwrite(A.f1.data(), 8, N_FINE, f);
    fwrite(A.f2.data(), 8, N_FINE, f);
    fwrite(A.t0.data(), 8, N_TAIL, f);
    fwrite(A.t1.data(), 8, N_TAIL, f);
    fwrite(A.t2.data(), 8, N_TAIL, f);
    fwrite(A.lg.data(), 8, N_LOG, f);
  }
  fclose(f);
  if (nfail) fprintf(stderr, "%s: %ld failed events\n", fout.c_str(), nfail);
  return 0;
}
