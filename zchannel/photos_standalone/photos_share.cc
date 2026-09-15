// Standalone Photos++ 3.61 measurement of the FSR energy sharing BETWEEN the
// two muon legs of Z -> mu+ mu-.
//
// Event generation is `photos_gen.cc`'s, verbatim: q qbar -> Z -> mu+ mu- at a
// pre-FSR mass drawn from the UL16 DY sample's own m_pre distribution inside a
// chosen band, Born (1+cos^2) + 2R cos angular distribution, the same HEPEVT
// construction and the same Photos switches.  Only the accumulator differs: it
// records how the loss is split over the two legs rather than the pair-mass
// kernel.
//
// In the PRE-FSR Z rest frame both muons start at E = m/2, so
//     x_q = 2 E'_q / m ,   z = m_post^2 / m^2 ,   u = -ln(m_post/m) ,
//     f   = (1 - x_+) / (1 - z)        [ + = mu+, pdgId -13 ]
// and a single massless photon puts the event exactly on the line
// x_+ + x_- = 1 + z, whose endpoints f = 0, 1 are the collinear configurations.
// The exact O(alpha) density is symmetric under f -> 1 - f, so the charge
// labelling is immaterial.  x_q is built as 2 (p'_q . Q_pre)/m^2 with Q_pre the
// PRE-FSR Z four-vector, which is the rest frame in which the starting energies
// are m/2 -- with --kinboost the lab energies are not, and boosting back with
// the post-FSR system would mix the sharing into the boost.
//
// Photos++ is a singleton with static state and is NOT thread safe, so
// parallelism is by separate processes, one per (band, replica);
// `share_merge.py` sums the outputs.

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

// --------------------------------------------------------------- constants --
static const double M_MU   = 0.1056583745;
static const double MZ     = 91.1876;
static const double GZ     = 2.4952;
static const double S2W    = 0.23153999447822571;   // the sample's POWHEG value
static const double NORAD_U = 1e-5;                 // the kernel's noise floor

// ---------------------------------------------------------------- binning --
// per-leg loss u_q = -ln x_q: bin 0 is "this leg lost nothing resolvable",
// bins 1..NUB-1 are geometric over [UJ_LO, UJ_HI] with the overflow clamped in.
static const int    NUB   = 91;
static const double UJ_LO = 1e-7, UJ_HI = 10.0;

// the u slices the sharing is quoted in
static const int    NSL = 5;
static const double USL[NSL + 1] = { 1e-4, 1e-3, 1e-2, 0.05, 0.2, 0.6 };

// the f grid of cmp_perleg.fig_share: geomspace(1e-6, 0.5, 25) mirrored about
// 1/2, so the binning is itself symmetric under f -> 1 - f
static const int NFE = 49, NFB = NFE - 1;
static double g_fedge[NFE];
static void build_fedge() {
  const int n = 25;
  const double lo = 1e-6, hi = 0.5;
  for (int i = 0; i < n; ++i)
    g_fedge[i] = lo * std::pow(hi / lo, (double)i / (double)(n - 1));
  g_fedge[0] = lo; g_fedge[n - 1] = hi;
  for (int i = 1; i < n; ++i) g_fedge[n - 1 + i] = 1.0 - g_fedge[n - 1 - i];
}

// the t grid the per-leg tails and the leg-leg correlation are quoted on
static const int    NT = 6;
static const double TCUT[NT] = { 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.2 };

// nph_class: 0 = exactly one photon and no pair (what Photos actually emitted),
// 1 = every event, 2 = the radiating events the LINE CONSTRAINT holds on,
// |k^2|/m^2 = |x_+ + x_- - (1+z)| < KTAG -- which is the tag the sample's own
// record has to use, having no reliable photon count, and which is NOT the same
// class: two photons collinear with each other also have k^2 ~ 0.
static const int NCL = 3;
static const double KTAG = 1e-8;      // --ktag overrides it
static double g_ktag = KTAG;

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
  // per-band scalars
  double n = 0, n_1g = 0, n_1g_rad = 0, n_rad = 0, n_rad6 = 0, n_viol = 0,
         n_bad = 0;
  double mpre_s1 = 0, sum_u = 0, sum_up = 0, sum_um = 0;
  // per class
  double cn[NCL] = { 0, 0, 0 }, cup[NCL] = { 0, 0, 0 }, cum[NCL] = { 0, 0, 0 };
  double tp[NCL][NT], tm[NCL][NT], tj[NCL][NT];
  // per (class, slice): count, count with 0.01 < f < 0.99, sum u, u^2, z, m
  double sn[NCL][NSL], smid[NCL][NSL], su[NCL][NSL], su2[NCL][NSL],
         sz[NCL][NSL], sm[NCL][NSL];
  std::vector<double> hj, hf;   // [NCL][NUB][NUB], [NCL][NSL][NFB]
  Acc() : hj((size_t)NCL * NUB * NUB, 0.0), hf((size_t)NCL * NSL * NFB, 0.0) {
    std::memset(tp, 0, sizeof tp); std::memset(tm, 0, sizeof tm);
    std::memset(tj, 0, sizeof tj);
    std::memset(sn, 0, sizeof sn); std::memset(smid, 0, sizeof smid);
    std::memset(su, 0, sizeof su); std::memset(su2, 0, sizeof su2);
    std::memset(sz, 0, sizeof sz); std::memset(sm, 0, sizeof sm);
  }

  static inline int ubin(double u) {
    if (!(u >= UJ_LO)) return 0;
    double t = std::log(u / UJ_LO) / std::log(UJ_HI / UJ_LO) * (double)(NUB - 1);
    int j = 1 + (int)t;
    if (j < 1) j = 1;
    if (j > NUB - 1) j = NUB - 1;
    return j;
  }
  static inline int fbin(double f) {
    // clip into the grid, symmetrically under f -> 1 - f, so that the first
    // and last bins are the two collinear overflows and nothing else
    double flo = g_fedge[0] * 1.001, fhi = 1.0 - flo;
    if (!(f > flo)) f = flo;
    if (f >= fhi) f = fhi;
    int j = (int)(std::upper_bound(g_fedge, g_fedge + NFE, f) - g_fedge) - 1;
    if (j < 0) j = 0;
    if (j > NFB - 1) j = NFB - 1;
    return j;
  }

  void fill(double m, double z, double u, double up, double um, double f,
            bool one_g, bool viol, bool ktag) {
    n += 1.0; mpre_s1 += m;
    sum_u += u; sum_up += up; sum_um += um;
    bool rad = u > NORAD_U;
    if (rad) n_rad += 1.0;
    if (u > 1e-6) n_rad6 += 1.0;      // the sample's own radiating definition
    if (one_g) { n_1g += 1.0; if (rad) n_1g_rad += 1.0; }
    if (viol) n_viol += 1.0;

    int islice = -1;
    for (int s = 0; s < NSL; ++s) if (u >= USL[s] && u < USL[s + 1]) { islice = s; break; }
    int jp = ubin(up), jm = ubin(um);
    int jf = (islice >= 0) ? fbin(f) : -1;
    bool mid = (f > 0.01 && f < 0.99);

    for (int c = 0; c < NCL; ++c) {
      if (c == 0 && !one_g) continue;
      if (c == 2 && !(ktag && rad)) continue;
      cn[c] += 1.0; cup[c] += up; cum[c] += um;
      for (int k = 0; k < NT; ++k) {
        bool bp = up > TCUT[k], bm = um > TCUT[k];
        if (bp) tp[c][k] += 1.0;
        if (bm) tm[c][k] += 1.0;
        if (bp && bm) tj[c][k] += 1.0;
      }
      hj[((size_t)c * NUB + jp) * NUB + jm] += 1.0;
      if (islice >= 0) {
        sn[c][islice] += 1.0;
        if (mid) smid[c][islice] += 1.0;
        su[c][islice] += u; su2[c][islice] += u * u;
        sz[c][islice] += z; sm[c][islice] += m;
        hf[((size_t)c * NSL + islice) * NFB + jf] += 1.0;
      }
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
  std::string fout   = arg_str(argc, argv, "--out",   "share.bin");
  long nev    = arg_i(argc, argv, "--n", 100000);
  int  b0     = (int)arg_i(argc, argv, "--band-first", 20);
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
  g_ktag = arg_d(argc, argv, "--ktag", KTAG);

  AngMode ang = sang == "flat" ? ANG_FLAT : (sang == "asym" ? ANG_ASYM : ANG_BORN);
  int qfix = squark == "u" ? 2 : (squark == "d" ? 1 : 0);

  build_fedge();
  Bands B; read_bands(fbands.c_str(), B);
  int nb = (int)B.lo.size();
  if (b1 < 0 || b1 >= nb) b1 = b0;
  BoostSample BS; if (useboost) BS.read(fboost.c_str());

  g_rng.seed((uint64_t)seed * 6364136223846793005ULL + 1442695040888963407ULL);

  // ---- Photos configuration -------------------------------------------
  // Photos::initialize() re-applies setExponentiation(true) (which resets
  // xphcut to 1e-7, isec/itre to 0 and the kinematic corrections to 5) and
  // unconditionally re-applies maxWtInterference(2.0) when interference is on,
  // so everything it touches has to be set again afterwards.  phokey.fint is an
  // over-sampling envelope, not a physics parameter.
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
            "fint=%g expeps=%g me=%d pairs=%d phot=%d ang=%s quark=%s boost=%d "
            "ktag=%.3e\n",
            phokey.iexp, phokey.interf, phokey.isec, phokey.itre,
            phocop.xphcut, phocop.alpha, phokey.fint, phokey.expeps,
            me, pairs, phot, sang.c_str(), squark.c_str(), useboost, g_ktag);
  }

  std::vector<Acc> acc(b1 - b0 + 1);
  long nfail = 0;

  for (int ib = b0; ib <= b1; ++ib) {
    MassSampler MS; MS.init(B, ib);
    Acc &A = acc[ib - b0];
    for (long iev = 0; iev < nev; ++iev) {
      double m = MS.draw(), ptz = 0.0, yz = 0.0;
      if (useboost) {
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

      // the pre-FSR Z, taken from the muons themselves so that the frame in
      // which each starts at m/2 is the one the boost-back uses
      double Q[4] = { mum[0] + mup[0], mum[1] + mup[1],
                      mum[2] + mup[2], mum[3] + mup[3] };

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

      if (p1->getPdgID() != 13 || p2->getPdgID() != -13) { ++nfail; ++A.n_bad; continue; }
      double ex = p1->getPx() + p2->getPx(), ey = p1->getPy() + p2->getPy();
      double ez = p1->getPz() + p2->getPz(), ee = p1->getE() + p2->getE();
      double m2 = ee * ee - ex * ex - ey * ey - ez * ez;
      if (!(m2 > 0.0) || !(m2 <= m * m * (1.0 + 1e-9))) {
        // m2 > m^2 can only be round-off at u = 0; anything else is a failure
        if (m2 > 0.0 && m2 < m * m * (1.0 + 1e-6)) m2 = m * m;
        else { ++nfail; ++A.n_bad; continue; }
      }
      double mpost = std::sqrt(m2);
      double u = -std::log(std::min(mpost / m, 1.0));
      double z = std::min(m2 / (m * m), 1.0);

      // x_q = 2 (p'_q . Q_pre) / m^2: the muon energy fraction in the PRE-FSR
      // Z rest frame, which is a plain invariant and needs no explicit boost
      double xp = 2.0 * (p2->getE() * Q[3] - p2->getPx() * Q[0]
                         - p2->getPy() * Q[1] - p2->getPz() * Q[2]) / (m * m);
      double xm = 2.0 * (p1->getE() * Q[3] - p1->getPx() * Q[0]
                         - p1->getPy() * Q[1] - p1->getPz() * Q[2]) / (m * m);
      double up = -std::log(std::min(std::max(xp, 1e-300), 1.0));
      double um = -std::log(std::min(std::max(xm, 1e-300), 1.0));
      // 1 - z = 0 only when nothing was emitted, and then there is no sharing
      double f = (1.0 - z > 0.0) ? (1.0 - xp) / (1.0 - z) : 0.5;
      // x_+ + x_- = (1+z) - k^2/m^2 exactly, with k the emitted system: the
      // line holds for one massless photon, and the violation is the emitted
      // system's own invariant mass, i.e. genuine multi-particle emission
      bool viol = std::fabs(xp + xm - (1.0 + z)) > 1e-9;
      bool ktag = std::fabs(xp + xm - (1.0 + z)) < g_ktag;

      int nph_add = 0, npair_add = 0;
      for (int k = n0; k < n1; ++k) {
        int id = evt.getParticle(k)->getPdgID();
        if (id == 22) ++nph_add; else ++npair_add;
      }
      A.fill(m, z, u, up, um, f, nph_add == 1 && npair_add == 0, viol, ktag);
    }
    if (verbose)
      fprintf(stderr, "band %d [%g,%g): n = %g, <u> = %.6e, 1g/rad = %.6f, "
              "P(mid|1g, u in [1e-2,0.05)) = %.6f\n",
              ib, B.lo[ib], B.hi[ib], A.n, A.sum_u / A.n,
              A.n_1g_rad / std::max(1.0, A.n_rad),
              A.smid[0][2] / std::max(1.0, A.sn[0][2]));
  }

  // ---- write ------------------------------------------------------------
  FILE *f = fopen(fout.c_str(), "wb");
  if (!f) { fprintf(stderr, "cannot write %s\n", fout.c_str()); exit(1); }
  char magic[8] = "PHSHR02";
  fwrite(magic, 1, 8, f);
  int64_t hdr[7] = { (int64_t)(b1 - b0 + 1), NCL, NUB, NSL, NFB, NT, NFE };
  fwrite(hdr, 8, 7, f);
  double hd[2] = { UJ_LO, UJ_HI };
  fwrite(hd, 8, 2, f);
  fwrite(USL, 8, NSL + 1, f);
  fwrite(TCUT, 8, NT, f);
  fwrite(g_fedge, 8, NFE, f);
  for (int ib = b0; ib <= b1; ++ib) {
    const Acc &A = acc[ib - b0];
    int64_t idx = ib;
    fwrite(&idx, 8, 1, f);
    double d[2] = { B.lo[ib], B.hi[ib] };
    fwrite(d, 8, 2, f);
    double s[11] = { A.n, A.n_1g, A.n_1g_rad, A.n_rad, A.n_rad6, A.n_viol,
                     A.n_bad, A.mpre_s1, A.sum_u, A.sum_up, A.sum_um };
    fwrite(s, 8, 11, f);
    fwrite(A.cn, 8, NCL, f);
    fwrite(A.cup, 8, NCL, f);
    fwrite(A.cum, 8, NCL, f);
    fwrite(A.tp, 8, NCL * NT, f);
    fwrite(A.tm, 8, NCL * NT, f);
    fwrite(A.tj, 8, NCL * NT, f);
    fwrite(A.sn, 8, NCL * NSL, f);
    fwrite(A.smid, 8, NCL * NSL, f);
    fwrite(A.su, 8, NCL * NSL, f);
    fwrite(A.su2, 8, NCL * NSL, f);
    fwrite(A.sz, 8, NCL * NSL, f);
    fwrite(A.sm, 8, NCL * NSL, f);
    fwrite(A.hj.data(), 8, A.hj.size(), f);
    fwrite(A.hf.data(), 8, A.hf.size(), f);
  }
  fclose(f);
  if (nfail) fprintf(stderr, "%s: %ld failed events\n", fout.c_str(), nfail);
  return 0;
}
