// Diagnostics of what Photos++ 3.61's real lepton-PAIR emission actually
// generates, for Z -> mu+ mu- at a FIXED pre-FSR mass in the Z rest frame.
//
// This is `photos_gen.cc` stripped of the band/boost machinery and instrumented
// on the emitted pair itself: for every event with a pair it records the pair
// invariant mass q, the pair energy fraction x_E = 2 E_pair / m, the event's
// mass loss u = -ln(m_post/m_pre), their correlation, and the angle of the pair
// w.r.t. the pre-FSR mu- direction.
//
// Photos++ is a singleton with static state and is NOT thread safe, so
// parallelism is by separate processes; `pairdiag_merge.py` sums the outputs.
//
// Built against the instrumented Photos copy (-DPAIRDIAG_INSTRUMENTED) it also
// dumps the per-call `trypar` bookkeeping (#PD... lines on stdout).

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
#ifdef PAIRDIAG_INSTRUMENTED
#include "Photos/pairs.h"
#endif

using namespace Photospp;

// --------------------------------------------------------------- constants --
static const double M_MU = 0.1056583745;
static const double MZ   = 91.1876;
static const double GZ   = 2.4952;
static const double S2W  = 0.23153999447822571;

// ---------------------------------------------------------------- binning --
static const int NQ = 200, NX = 200, NU = 200, NC = 100, N2 = 40;
static const double Q_LO = 1e-3,  Q_HI = 1e2;     // pair mass          [GeV]
static const double X_LO = 1e-6,  X_HI = 2.0;     // x_E = 2 E_pair / m
static const double U_LO = 1e-9,  U_HI = 2.0;     // u = -ln(m_post/m)
static const double Q2_LO = 1e-3, Q2_HI = 1e2;    // 2-D q axis
static const double U2_LO = 1e-9, U2_HI = 2.0;    // 2-D u axis

static const double QCUT[6] = { 2.0 * 0.000511, 0.01, 0.1, 1.0, 3.0, 10.0 };
static const double XCUT[5] = { 1e-4, 1e-3, 1e-2, 0.1, 0.5 };

// per-species scalar block
enum { S_N = 0, S_SUMU, S_SUMQ, S_SUMXE, S_SUMX,
       S_NQ = 5, S_SUMUQ = 11, S_NX = 17,
       S_COSP = 22, S_COSM, S_COSPHI, S_COSMHI,
       S_SUMR = 26, S_SUMABSR, S_SUMRELR, S_SUMRELR2, S_NR,
       S_NSCAL = 31 };
// global scalar block
enum { G_NEV = 0, G_NEVPAIR, G_SUMU, G_SUMU2, G_SUMX, G_NUPOS, G_NADD_OTHER,
       G_NBAD, G_NSCAL = 8 };

static const int NHIST = NQ + NX + NU + NC + N2 * N2;
static const int NSUM  = G_NSCAL + 2 * (S_NSCAL + NHIST);
static const int NMIN  = 4;     // {minq, minx} x species
static const int NMAX  = 6;     // {maxq, maxx, max|resid|} x species

static std::vector<double> g_sum(NSUM, 0.0);
static double g_min[NMIN] = { 1e300, 1e300, 1e300, 1e300 };
static double g_max[NMAX] = { -1e300, -1e300, -1e300, -1e300, -1e300, -1e300 };

static inline double *spec(int s)  { return &g_sum[G_NSCAL + s * (S_NSCAL + NHIST)]; }
static inline double *hist(int s)  { return spec(s) + S_NSCAL; }

static inline void logfill(double *h, int n, double lo, double hi, double v, double w) {
  if (!(v > 0.0)) return;
  double t = std::log(v / lo) / std::log(hi / lo) * (double)n;
  int j = (int)t;
  if (j < 0) j = 0;
  if (j >= n) j = n - 1;
  h[j] += w;
}

// ------------------------------------------------------------------- rng ----
static std::mt19937_64 g_rng;
static inline double urand() {
  for (;;) {
    double u = std::generate_canonical<double, 53>(g_rng);
    if (u > 0.0 && u < 1.0) return u;
  }
}
static double photos_rand() { return urand(); }

// ------------------------------------------------------- angular sampling --
static double afb_ratio(double s, int quark) {
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
static double draw_cos(double R) {
  double fmax = 2.0 + 2.0 * std::fabs(R);
  for (;;) {
    double c = 2.0 * urand() - 1.0;
    double f = (1.0 + c * c) + 2.0 * R * c;
    if (f > 0.0 && urand() * fmax < f) return c;
  }
}

// ------------------------------------------------------------------- args --
static std::string arg_str(int argc, char **argv, const char *k, const char *def) {
  size_t L = strlen(k);
  for (int i = 1; i < argc; ++i)
    if (!strncmp(argv[i], k, L) && argv[i][L] == '=') return std::string(argv[i] + L + 1);
  return std::string(def);
}
static double arg_d(int argc, char **argv, const char *k, double d) {
  std::string s = arg_str(argc, argv, k, "");  return s.empty() ? d : atof(s.c_str());
}
static long arg_i(int argc, char **argv, const char *k, long d) {
  std::string s = arg_str(argc, argv, k, "");  return s.empty() ? d : atol(s.c_str());
}

// ---------------------------------------------------- optional m_pre input --
// The sample's own m_pre density (the `mpre_bands.bin` written by prep_input.py),
// used whole rather than band by band, so that the pair rate can be quoted for
// the sample's mass distribution instead of a fixed mass.
struct MpreSampler {
  std::vector<double> edges, cdf;
  bool active = false;
  void read(const char *fn) {
    FILE *f = fopen(fn, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", fn); exit(1); }
    int32_t nb = 0;
    if (fread(&nb, 4, 1, f) != 1) exit(1);
    if (fseek(f, (long)nb * (3 * 8 + 2 * 8), SEEK_CUR)) exit(1);
    int64_t ne = 0;
    if (fread(&ne, 8, 1, f) != 1) exit(1);
    edges.resize(ne);
    std::vector<double> h(ne - 1);
    if ((int64_t)fread(edges.data(), 8, ne, f) != ne) exit(1);
    if ((int64_t)fread(h.data(), 8, ne - 1, f) != ne - 1) exit(1);
    fclose(f);
    cdf.assign(h.size(), 0.0);
    double s = 0.0;
    for (size_t k = 0; k < h.size(); ++k) { s += std::max(0.0, h[k]); cdf[k] = s; }
    for (size_t k = 0; k < cdf.size(); ++k) cdf[k] /= s;
    active = true;
  }
  double draw() const {
    double r = urand();
    size_t k = std::lower_bound(cdf.begin(), cdf.end(), r) - cdf.begin();
    if (k >= cdf.size()) k = cdf.size() - 1;
    return edges[k] + (edges[k + 1] - edges[k]) * urand();
  }
};

// ------------------------------------------------------------------- main --
int main(int argc, char **argv) {
  std::string fout = arg_str(argc, argv, "--out", "pairdiag.bin");
  std::string fmpre = arg_str(argc, argv, "--mpre", "");
  long   nev   = arg_i(argc, argv, "--n", 1000000);
  long   seed  = arg_i(argc, argv, "--seed", 1);
  double mfix  = arg_d(argc, argv, "--mass", MZ);
  int    pairs = (int)arg_i(argc, argv, "--pairs", 1);
  int    phot  = (int)arg_i(argc, argv, "--phot", 0);
  int    expo  = (int)arg_i(argc, argv, "--exp", 1);
  int    me    = (int)arg_i(argc, argv, "--me", 0);
  double fint  = arg_d(argc, argv, "--fint", 0.0);
  double xph   = arg_d(argc, argv, "--xphcut", 1e-7);
  double fracu = arg_d(argc, argv, "--frac-up", 0.6);
  int    verbose = (int)arg_i(argc, argv, "--verbose", 0);

  MpreSampler MS;
  if (!fmpre.empty()) MS.read(fmpre.c_str());

  g_rng.seed((uint64_t)seed * 6364136223846793005ULL + 1442695040888963407ULL);

  Photos::setMomentumUnit(Photos::GEV);
  Photos::setRandomGenerator(photos_rand);
  Photos::setExponentiation(expo != 0);
  Photos::setMeCorrectionWtForZ(me != 0);
  Photos::setPairEmission(pairs != 0);
  Photos::setPhotonEmission(phot != 0);
  Photos::setStopAtCriticalError(false);
  Photos::initialize();
  Photos::setInfraredCutOff(xph);
  if (fint > 0.0) Photos::maxWtInterference(fint);

  for (long iev = 0; iev < nev; ++iev) {
    double m = MS.active ? MS.draw() : mfix;
    if (m <= 2.0 * M_MU) continue;
    int quark = (urand() < fracu) ? 2 : 1;
    double R = afb_ratio(m * m, quark);
    double c = draw_cos(R), s = std::sqrt(std::max(0.0, 1.0 - c * c));
    double ph = 2.0 * M_PI * urand();
    double E = 0.5 * m, p = std::sqrt(std::max(0.0, E * E - M_MU * M_MU));

    double mum[4] = { p * s * std::cos(ph), p * s * std::sin(ph), p * c, E };
    double mup[4] = { -mum[0], -mum[1], -mum[2], E };
    // the pre-FSR mu- direction, kept to tell which leg the pair follows
    double n0v[3] = { mum[0] / p, mum[1] / p, mum[2] / p };

    int qid = (quark == 2) ? 2 : 1;
    PhotosHEPEVTEvent evt;
    evt.addParticle(new PhotosHEPEVTParticle( qid, PhotosParticle::HISTORY,
                    0.0, 0.0,  0.5 * m, 0.5 * m, 0.0, -1, -1, 2, 2));
    evt.addParticle(new PhotosHEPEVTParticle(-qid, PhotosParticle::HISTORY,
                    0.0, 0.0, -0.5 * m, 0.5 * m, 0.0, -1, -1, 2, 2));
    evt.addParticle(new PhotosHEPEVTParticle(23, PhotosParticle::DECAYED,
                    0.0, 0.0, 0.0, m, m, 0, 1, 3, 4));
    PhotosHEPEVTParticle *p1 = new PhotosHEPEVTParticle(13, PhotosParticle::STABLE,
                    mum[0], mum[1], mum[2], mum[3], M_MU, 2, -1, -1, -1);
    PhotosHEPEVTParticle *p2 = new PhotosHEPEVTParticle(-13, PhotosParticle::STABLE,
                    mup[0], mup[1], mup[2], mup[3], M_MU, 2, -1, -1, -1);
    evt.addParticle(p1);
    evt.addParticle(p2);

    int na = evt.getParticleCount();
    evt.process();
    int nb = evt.getParticleCount();

    g_sum[G_NEV] += 1.0;

    if (p1->getPdgID() != 13 || p2->getPdgID() != -13) { g_sum[G_NBAD] += 1.0; continue; }
    double ex = p1->getPx() + p2->getPx(), ey = p1->getPy() + p2->getPy();
    double ez = p1->getPz() + p2->getPz(), ee = p1->getE() + p2->getE();
    double m2 = ee * ee - ex * ex - ey * ey - ez * ez;
    if (!(m2 > 0.0) || !(m2 <= m * m * (1.0 + 1e-9))) {
      if (m2 > 0.0 && m2 < m * m * (1.0 + 1e-6)) m2 = m * m;
      else { g_sum[G_NBAD] += 1.0; continue; }
    }
    double u = -std::log(std::min(std::sqrt(m2) / m, 1.0));
    double x = -std::expm1(-2.0 * u);
    g_sum[G_SUMU] += u;
    g_sum[G_SUMU2] += u * u;
    g_sum[G_SUMX] += x;
    if (u > 0.0) g_sum[G_NUPOS] += 1.0;

    // ---- the emitted pair --------------------------------------------
    double pv[4] = { 0, 0, 0, 0 };
    int nlep = 0, sp = -1;
    for (int k = na; k < nb; ++k) {
      PhotosParticle *pp = evt.getParticle(k);
      int id = std::abs(pp->getPdgID());
      if (id == 11 || id == 13) {
        ++nlep;
        sp = (id == 11) ? 0 : 1;
        pv[0] += pp->getPx(); pv[1] += pp->getPy();
        pv[2] += pp->getPz(); pv[3] += pp->getE();
      } else if (id != 22) {
        g_sum[G_NADD_OTHER] += 1.0;
      }
    }
    if (nlep == 0) continue;
    g_sum[G_NEVPAIR] += 1.0;
    if (nlep != 2) continue;      // >1 pair: not expected, excluded from the shapes

    double q2 = pv[3] * pv[3] - pv[0] * pv[0] - pv[1] * pv[1] - pv[2] * pv[2];
    double q  = std::sqrt(std::max(0.0, q2));
    double xE = 2.0 * pv[3] / m;
    double pm = std::sqrt(pv[0] * pv[0] + pv[1] * pv[1] + pv[2] * pv[2]);
    double ct = (pm > 0.0) ? (pv[0] * n0v[0] + pv[1] * n0v[1] + pv[2] * n0v[2]) / pm : 0.0;

    double *S = spec(sp), *H = hist(sp);
    S[S_N]    += 1.0;
    S[S_SUMU] += u;
    S[S_SUMQ] += q;
    S[S_SUMXE] += xE;
    S[S_SUMX] += x;
    for (int k = 0; k < 6; ++k) if (q > QCUT[k]) { S[S_NQ + k] += 1.0; S[S_SUMUQ + k] += u; }
    for (int k = 0; k < 5; ++k) if (xE > XCUT[k]) S[S_NX + k] += 1.0;
    if (ct > 0.0) { S[S_COSP] += 1.0; if (ct >  0.9) S[S_COSPHI] += 1.0; }
    else          { S[S_COSM] += 1.0; if (ct < -0.9) S[S_COSMHI] += 1.0; }

    // closure of u against (x_E, q): 1 - exp(-2u) = x_E - q^2/m^2
    double pred = xE - q2 / (m * m);
    double r = x - pred;
    S[S_SUMR] += r; S[S_SUMABSR] += std::fabs(r); S[S_NR] += 1.0;
    if (pred != 0.0) { double rr = r / pred; S[S_SUMRELR] += rr; S[S_SUMRELR2] += rr * rr; }
    if (std::fabs(r) > g_max[3 * sp + 2]) g_max[3 * sp + 2] = std::fabs(r);

    if (q  < g_min[2 * sp + 0]) g_min[2 * sp + 0] = q;
    if (xE < g_min[2 * sp + 1]) g_min[2 * sp + 1] = xE;
    if (q  > g_max[3 * sp + 0]) g_max[3 * sp + 0] = q;
    if (xE > g_max[3 * sp + 1]) g_max[3 * sp + 1] = xE;

    logfill(H,                NQ, Q_LO, Q_HI, q,  1.0);
    logfill(H + NQ,           NX, X_LO, X_HI, xE, 1.0);
    logfill(H + NQ + NX,      NU, U_LO, U_HI, u,  1.0);
    int jc = (int)((ct + 1.0) * 0.5 * NC);
    if (jc < 0) jc = 0;
    if (jc >= NC) jc = NC - 1;
    H[NQ + NX + NU + jc] += 1.0;
    if (q > 0.0 && u > 0.0) {
      double tq = std::log(q / Q2_LO) / std::log(Q2_HI / Q2_LO) * N2;
      double tu = std::log(u / U2_LO) / std::log(U2_HI / U2_LO) * N2;
      int jq = (int)tq, ju = (int)tu;
      if (jq < 0) jq = 0; if (jq >= N2) jq = N2 - 1;
      if (ju < 0) ju = 0; if (ju >= N2) ju = N2 - 1;
      H[NQ + NX + NU + NC + ju * N2 + jq] += 1.0;
    }
  }

  FILE *f = fopen(fout.c_str(), "wb");
  if (!f) { fprintf(stderr, "cannot write %s\n", fout.c_str()); exit(1); }
  char magic[8] = "PDIAG02";
  fwrite(magic, 1, 8, f);
  int64_t hdr[8] = { NQ, NX, NU, N2, NC, NSUM, NMIN, NMAX };
  fwrite(hdr, 8, 8, f);
  fwrite(g_sum.data(), 8, NSUM, f);
  fwrite(g_min, 8, NMIN, f);
  fwrite(g_max, 8, NMAX, f);
  fclose(f);

  if (verbose) {
    double n = g_sum[G_NEV];
    fprintf(stderr, "n = %.0f, pairs = %.0f (%.6e/ev), <u> = %.6e\n",
            n, spec(0)[S_N] + spec(1)[S_N],
            (spec(0)[S_N] + spec(1)[S_N]) / n, g_sum[G_SUMU] / n);
  }
#ifdef PAIRDIAG_INSTRUMENTED
  photos_pair_diag_dump();
#endif
  return 0;
}
