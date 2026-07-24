// Pixel Lorentz-angle response study: per-hit tree of (refitted, on-track
// template-CPE) BPix hit positions vs matched muon PSimHit truth, with the
// cluster pathology classification used by the CVH pixel-hit study.
// Run behind a TrackRefitter so the hit positions are re-evaluated with
// the track angle (the estimator used in the CVH refit).

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"

#include "TTree.h"

class PixelLAResponse : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit PixelLAResponse(const edm::ParameterSet& iConfig)
      : assoToken_(consumes<TrajTrackAssociationCollection>(
            iConfig.getParameter<edm::InputTag>("trajTrackAsso"))),
        simHitToken_(consumes<std::vector<PSimHit>>(
            edm::InputTag("g4SimHits", "TrackerHitsPixelBarrelLowTof"))) {
    usesResource("TFileService");
    edm::Service<TFileService> fs;
    tree_ = fs->make<TTree>("hits", "hits");
    tree_->Branch("det", &det_);
    tree_->Branch("layer", &layer_);
    tree_->Branch("cls", &cls_);
    tree_->Branch("sizeX", &sizeX_);
    tree_->Branch("dx", &dx_);
    tree_->Branch("dy", &dy_);
    tree_->Branch("lx", &lx_);
    tree_->Branch("tanax", &tanax_);
  }

private:
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override {
    edm::Handle<TrajTrackAssociationCollection> assoH;
    iEvent.getByToken(assoToken_, assoH);
    edm::Handle<std::vector<PSimHit>> simH;
    iEvent.getByToken(simHitToken_, simH);

    // muon simhits per detid
    std::map<unsigned int, std::vector<const PSimHit*>> simbydet;
    for (const auto& sh : *simH) {
      if (std::abs(sh.particleType()) == 13) {
        simbydet[sh.detUnitId()].push_back(&sh);
      }
    }

    for (const auto& asso : *assoH) {
      const Trajectory& traj = *asso.key;
      for (const auto& meas : traj.measurements()) {
        const auto& hit = meas.recHit();
        if (!hit->isValid()) continue;
        const DetId detid = hit->geographicalId();
        if (detid.subdetId() != PixelSubdetector::PixelBarrel) continue;
        const SiPixelRecHit* ph = dynamic_cast<const SiPixelRecHit*>(hit->hit());
        if (ph == nullptr || ph->cluster().isNull()) continue;
        const SiPixelCluster& cl = *ph->cluster();
        auto it = simbydet.find(detid.rawId());
        if (it == simbydet.end()) continue;

        const LocalPoint lp = hit->localPosition();
        const PSimHit* best = nullptr;
        double bestd = 0.05;  // 500 um match window
        for (const PSimHit* sh : it->second) {
          const double d = std::abs(sh->localPosition().x() - lp.x());
          if (d < bestd) { best = sh; bestd = d; }
        }
        if (best == nullptr) continue;

        const PixelGeomDetUnit* pdet =
            dynamic_cast<const PixelGeomDetUnit*>(hit->det());
        if (pdet == nullptr) continue;
        const PixelTopology& topo = pdet->specificTopology();

        int cls = 0;
        if (cl.minPixelRow() == 0) cls |= 1;
        if (cl.maxPixelRow() == topo.nrows() - 1) cls |= 2;
        if (cl.minPixelCol() == 0 || cl.maxPixelCol() == topo.ncolumns() - 1) cls |= 4;
        if (cl.sizeX() <= 1) cls |= 8;
        if (cl.sizeY() <= 1) cls |= 16;

        det_ = detid.rawId();
        layer_ = PXBDetId(detid).layer();
        cls_ = cls;
        sizeX_ = cl.sizeX();
        dx_ = lp.x() - best->localPosition().x();
        dy_ = lp.y() - best->localPosition().y();
        lx_ = lp.x();
        const auto& mom = best->momentumAtEntry();
        tanax_ = std::abs(mom.z()) > 1e-6 ? mom.x() / mom.z() : 999.f;
        tree_->Fill();
      }
    }
  }

  const edm::EDGetTokenT<TrajTrackAssociationCollection> assoToken_;
  const edm::EDGetTokenT<std::vector<PSimHit>> simHitToken_;
  TTree* tree_;
  unsigned int det_;
  int layer_, cls_, sizeX_;
  float dx_, dy_, lx_, tanax_;
};

DEFINE_FWK_MODULE(PixelLAResponse);
