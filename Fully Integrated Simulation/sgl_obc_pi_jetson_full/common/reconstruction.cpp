#include "reconstruction.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace fs = std::filesystem;
namespace sgl {
static void draw_grid_overlay(ImageRGBA& img,int tile_px_x,int tile_px_y){ unsigned w=img.w,h=img.h; for(unsigned y=0;y<h;y++) if(tile_px_y>0 && y%(unsigned)tile_px_y==0) for(unsigned x=0;x<w;x++){ size_t i=4ull*(y*w+x); img.rgba[i]=255; img.rgba[i+1]=0; img.rgba[i+2]=0; } for(unsigned x=0;x<w;x++) if(tile_px_x>0 && x%(unsigned)tile_px_x==0) for(unsigned y=0;y<h;y++){ size_t i=4ull*(y*w+x); img.rgba[i]=255; img.rgba[i+1]=0; img.rgba[i+2]=0; } }
std::vector<TileStat> compute_tiles(const ImageRGBA& src,int tile_px_x,int tile_px_y,int& tx,int& ty){ tx=std::max(1,(int)((src.w+tile_px_x-1)/tile_px_x)); ty=std::max(1,(int)((src.h+tile_px_y-1)/tile_px_y)); std::vector<TileStat> stats((size_t)tx*ty); 
#pragma omp parallel for collapse(2) schedule(static)
  for(int r=0;r<ty;r++) for(int c=0;c<tx;c++){ int x0=c*tile_px_x,y0=r*tile_px_y,x1=std::min((int)src.w,x0+tile_px_x),y1=std::min((int)src.h,y0+tile_px_y); double sr=0,sg=0,sb=0,sl=0; long long cnt=0; for(int y=y0;y<y1;y++) for(int x=x0;x<x1;x++){ const uint8_t* p=&src.rgba[4ull*(y*src.w+x)]; double R=p[0]/255.0,G=p[1]/255.0,B=p[2]/255.0,L=0.2126*R+0.7152*G+0.0722*B; sr+=R; sg+=G; sb+=B; sl+=L; cnt++; } if(cnt==0) cnt=1; stats[(size_t)r*tx+c]=TileStat{sr/cnt,sg/cnt,sb/cnt,sl/cnt}; } return stats; }
ImageRGBA render_ring_from_tiles(const std::vector<TileStat>& tiles,int tx,int ty,int N,double ring_radius,double ring_sigma){ ImageRGBA out; out.w=N; out.h=N; out.rgba.assign(4ull*N*N,255); const double xmin=-1.0,xmax=1.0,scale=(xmax-xmin)/N; 
#pragma omp parallel for collapse(2) schedule(static)
  for(int y=0;y<N;y++) for(int x=0;x<N;x++){ double X=xmin+(x+0.5)*scale,Y=xmin+(y+0.5)*scale; double r=std::sqrt(X*X+Y*Y),theta=std::atan2(Y,X); double dr=r-ring_radius,I=std::exp(-(dr*dr)/(2.0*ring_sigma*ring_sigma)); double u=(theta+M_PI)/(2.0*M_PI); int c=std::clamp((int)std::floor(u*tx),0,tx-1); double v=(Y-xmin)/(xmax-xmin); int rr=std::clamp((int)std::floor(v*ty),0,ty-1); const TileStat&t=tiles[(size_t)rr*tx+c]; size_t i=4ull*(y*N+x); out.rgba[i]=(uint8_t)std::lround(255.0*std::clamp(I*t.r,0.0,1.0)); out.rgba[i+1]=(uint8_t)std::lround(255.0*std::clamp(I*t.g,0.0,1.0)); out.rgba[i+2]=(uint8_t)std::lround(255.0*std::clamp(I*t.b,0.0,1.0)); out.rgba[i+3]=255; } return out; }
static void clamp_roi(proto::RegionOfInterest& r, int ctx, int cty) {
  if (ctx <= 0 || cty <= 0) {
    r = proto::RegionOfInterest{};
    return;
  }
  r.w = std::max(1, r.w);
  r.h = std::max(1, r.h);
  r.x = std::clamp(r.x, 0, ctx - 1);
  r.y = std::clamp(r.y, 0, cty - 1);
  r.w = std::min(r.w, ctx - r.x);
  r.h = std::min(r.h, cty - r.y);
}
static ImageRGBA reconstruct_grouped(const std::vector<TileStat>& tiles,int tx,int ty,int gx,int gy,unsigned outW,unsigned outH,std::vector<TileStat>* grouped_out){ gx=std::max(1,gx); gy=std::max(1,gy); int ctx=std::max(1,(tx+gx-1)/gx), cty=std::max(1,(ty+gy-1)/gy); std::vector<TileStat> grouped((size_t)ctx*cty); 
#pragma omp parallel for collapse(2) schedule(static)
  for(int r=0;r<cty;r++) for(int c=0;c<ctx;c++){ double sr=0,sg=0,sb=0,sl=0; long long cnt=0; for(int rr=r*gy; rr<std::min(ty,(r+1)*gy); ++rr) for(int cc=c*gx; cc<std::min(tx,(c+1)*gx); ++cc){ const TileStat&t=tiles[(size_t)rr*tx+cc]; sr+=t.r; sg+=t.g; sb+=t.b; sl+=t.l; cnt++; } if(cnt==0) cnt=1; grouped[(size_t)r*ctx+c]=TileStat{sr/cnt,sg/cnt,sb/cnt,sl/cnt}; }
  if(grouped_out) *grouped_out=grouped; ImageRGBA rec; rec.w=outW; rec.h=outH; rec.rgba.assign(4ull*outW*outH,255); unsigned cellW=(outW+(unsigned)ctx-1)/(unsigned)ctx, cellH=(outH+(unsigned)cty-1)/(unsigned)cty; 
#pragma omp parallel for collapse(2) schedule(static)
  for(int r=0;r<cty;r++) for(int c=0;c<ctx;c++){ const TileStat&t=grouped[(size_t)r*ctx+c]; unsigned x0=(unsigned)c*cellW,y0=(unsigned)r*cellH,x1=std::min(outW,x0+cellW),y1=std::min(outH,y0+cellH); for(unsigned y=y0;y<y1;y++) for(unsigned x=x0;x<x1;x++){ size_t i=4ull*(y*outW+x); rec.rgba[i]=(uint8_t)std::lround(255.0*t.r); rec.rgba[i+1]=(uint8_t)std::lround(255.0*t.g); rec.rgba[i+2]=(uint8_t)std::lround(255.0*t.b); rec.rgba[i+3]=255; } } return rec; }
ImageRGBA reconstruct_coarse_from_tiles(const std::vector<TileStat>& tiles,int tx,int ty,int coarse_groups_x,int coarse_groups_y,unsigned outW,unsigned outH,std::vector<proto::RegionOfInterest>* rois,int max_rois,const std::vector<proto::RegionOfInterest>* prior_rois,int prior_roi_growth,double* roi_selection_ms){ std::vector<TileStat> grouped; ImageRGBA rec=reconstruct_grouped(tiles,tx,ty,coarse_groups_x,coarse_groups_y,outW,outH,&grouped); if(!rois||max_rois<=0) return rec; const auto t0=std::chrono::steady_clock::now(); int gx=std::max(1,coarse_groups_x), gy=std::max(1,coarse_groups_y), ctx=std::max(1,(tx+gx-1)/gx), cty=std::max(1,(ty+gy-1)/gy); std::vector<double> prior_bias((size_t)ctx*cty,0.0); if(prior_rois && !prior_rois->empty()){ const int g=std::max(0,prior_roi_growth); for(const auto& p0:*prior_rois){ auto p=p0; p.x-=g; p.y-=g; p.w+=2*g; p.h+=2*g; clamp_roi(p,ctx,cty); for(int yy=p.y; yy<p.y+p.h; ++yy) for(int xx=p.x; xx<p.x+p.w; ++xx) prior_bias[(size_t)yy*ctx+xx]=1.0; } } std::vector<proto::RegionOfInterest> scored; scored.reserve((size_t)ctx*cty); for(int r=0;r<cty;r++) for(int c=0;c<ctx;c++){ const TileStat& mean=grouped[(size_t)r*ctx+c]; double color_var=0.0,lum_var=0.0,edge=0.0,contrast=0.0,bg_penalty=0.0; long long cnt=0; for(int rr=r*gy; rr<std::min(ty,(r+1)*gy); ++rr){ for(int cc=c*gx; cc<std::min(tx,(c+1)*gx); ++cc){ const TileStat&t=tiles[(size_t)rr*tx+cc]; color_var+=std::fabs(t.r-mean.r)+std::fabs(t.g-mean.g)+std::fabs(t.b-mean.b); double dl=t.l-mean.l; lum_var+=dl*dl; if(cc+1<std::min(tx,(c+1)*gx)){ const TileStat& tr=tiles[(size_t)rr*tx+(cc+1)]; edge+=std::fabs(t.l-tr.l); } if(rr+1<std::min(ty,(r+1)*gy)){ const TileStat& td=tiles[(size_t)(rr+1)*tx+cc]; edge+=std::fabs(t.l-td.l); } contrast+=std::fabs(t.l-0.5); if(t.l<0.05 || t.l>0.97) bg_penalty+=1.0; cnt++; } } if(cnt==0) cnt=1; double inv=1.0/(double)cnt; color_var*=inv; lum_var=std::sqrt(std::max(0.0,lum_var*inv)); edge*=inv; contrast*=inv; bg_penalty*=inv; double prior=prior_bias[(size_t)r*ctx+c]; double s=1.20*edge + 0.95*lum_var + 0.75*color_var + 0.45*contrast + 0.60*prior - 0.90*bg_penalty; if(s<0.0) s=0.0; scored.push_back(proto::RegionOfInterest{c,r,1,1,s}); } std::sort(scored.begin(),scored.end(),[](auto&a,auto&b){return a.score>b.score;}); if((int)scored.size()>max_rois) scored.resize((size_t)max_rois); *rois=scored; if(roi_selection_ms){ const auto t1=std::chrono::steady_clock::now(); *roi_selection_ms = std::chrono::duration<double,std::milli>(t1-t0).count(); } return rec; }
ImageRGBA refine_from_tiles(const std::vector<TileStat>& tiles,int tx,int ty,int coarse_groups_x,int coarse_groups_y,const std::vector<proto::RegionOfInterest>& rois,unsigned outW,unsigned outH){ ImageRGBA rec=reconstruct_grouped(tiles,tx,ty,coarse_groups_x,coarse_groups_y,outW,outH,nullptr); int gx=std::max(1,coarse_groups_x), gy=std::max(1,coarse_groups_y), ctx=std::max(1,(tx+gx-1)/gx), cty=std::max(1,(ty+gy-1)/gy); unsigned coarseW=(outW+(unsigned)ctx-1)/(unsigned)ctx, coarseH=(outH+(unsigned)cty-1)/(unsigned)cty; for(const auto&roi:rois){ for(int rg_y=roi.y; rg_y<roi.y+roi.h; ++rg_y) for(int rg_x=roi.x; rg_x<roi.x+roi.w; ++rg_x){ if(rg_x<0||rg_x>=ctx||rg_y<0||rg_y>=cty) continue; int fx0=rg_x*gx, fy0=rg_y*gy, fx1=std::min(tx,fx0+gx), fy1=std::min(ty,fy0+gy); unsigned rx0=(unsigned)rg_x*coarseW, ry0=(unsigned)rg_y*coarseH, rx1=std::min(outW,rx0+coarseW), ry1=std::min(outH,ry0+coarseH); unsigned cellW=std::max(1u,(rx1-rx0+(unsigned)(fx1-fx0)-1)/(unsigned)(fx1-fx0)), cellH=std::max(1u,(ry1-ry0+(unsigned)(fy1-fy0)-1)/(unsigned)(fy1-fy0)); for(int fy=fy0; fy<fy1; ++fy) for(int fx=fx0; fx<fx1; ++fx){ const TileStat&t=tiles[(size_t)fy*tx+fx]; unsigned x0=rx0+(unsigned)(fx-fx0)*cellW,y0=ry0+(unsigned)(fy-fy0)*cellH,x1=std::min(rx1,x0+cellW),y1=std::min(ry1,y0+cellH); for(unsigned y=y0;y<y1;y++) for(unsigned x=x0;x<x1;x++){ size_t i=4ull*(y*outW+x); rec.rgba[i]=(uint8_t)std::lround(255.0*t.r); rec.rgba[i+1]=(uint8_t)std::lround(255.0*t.g); rec.rgba[i+2]=(uint8_t)std::lround(255.0*t.b); rec.rgba[i+3]=255; } } } } return rec; }
std::string tiles_to_csv(const std::vector<TileStat>& tiles,int tx,int ty){ std::ostringstream oss; oss<<"tx,"<<tx<<"\n"<<"ty,"<<ty<<"\n"<<"row,col,r,g,b,l\n"; for(int r=0;r<ty;r++) for(int c=0;c<tx;c++){ const auto&t=tiles[(size_t)r*tx+c]; oss<<r<<","<<c<<","<<t.r<<","<<t.g<<","<<t.b<<","<<t.l<<"\n"; } return oss.str(); }
bool tiles_from_csv(const std::string& csv,std::vector<TileStat>& tiles,int& tx,int& ty){ tx=ty=0; std::istringstream iss(csv); std::string line; std::getline(iss,line); if(line.rfind("tx,",0)!=0) return false; tx=std::stoi(line.substr(3)); std::getline(iss,line); if(line.rfind("ty,",0)!=0) return false; ty=std::stoi(line.substr(3)); std::getline(iss,line); tiles.assign((size_t)tx*ty,TileStat{}); while(std::getline(iss,line)){ if(line.empty()) continue; std::istringstream is(line); std::string tok; std::vector<std::string> p; while(std::getline(is,tok,',')) p.push_back(tok); if(p.size()<6) continue; int r=std::stoi(p[0]), c=std::stoi(p[1]); if(r>=0&&r<ty&&c>=0&&c<tx) tiles[(size_t)r*tx+c]=TileStat{std::stod(p[2]),std::stod(p[3]),std::stod(p[4]),std::stod(p[5])}; } return true; }
std::vector<uint8_t> ppm_bytes(const ImageRGBA& img){ std::ostringstream oss; oss<<"P6\n"<<img.w<<" "<<img.h<<"\n255\n"; std::string hdr=oss.str(); std::vector<uint8_t> out(hdr.begin(),hdr.end()); out.reserve(out.size()+img.w*img.h*3); for(size_t i=0;i<img.rgba.size()/4;i++){ out.push_back(img.rgba[4*i]); out.push_back(img.rgba[4*i+1]); out.push_back(img.rgba[4*i+2]); } return out; }
static ImageRGBA tiles_to_image(const std::vector<TileStat>& tiles,int tx,int ty){ ImageRGBA out; out.w=(unsigned)std::max(1,tx); out.h=(unsigned)std::max(1,ty); out.rgba.assign(4ull*out.w*out.h,255); for(int y=0;y<ty;y++) for(int x=0;x<tx;x++){ const auto&t=tiles[(size_t)y*tx+x]; size_t i=4ull*(y*out.w+x); out.rgba[i]=(uint8_t)std::lround(255.0*std::clamp(t.r,0.0,1.0)); out.rgba[i+1]=(uint8_t)std::lround(255.0*std::clamp(t.g,0.0,1.0)); out.rgba[i+2]=(uint8_t)std::lround(255.0*std::clamp(t.b,0.0,1.0)); } return out; }
static bool extract_tiles_from_ring_observation(const ImageRGBA& obs,int tx,int ty,std::vector<TileStat>& tiles,double& cx,double& cy,double& r_peak,double& r_sigma){ if(obs.w==0||obs.h==0) return false; tx=std::max(16,tx); ty=std::max(8,ty); std::vector<double> lum((size_t)obs.w*obs.h,0.0); double lmin=1e9,lmax=-1e9; for(size_t i=0;i<lum.size();++i){ const uint8_t* p=&obs.rgba[4*i]; double L=(0.2126*p[0]+0.7152*p[1]+0.0722*p[2])/255.0; lum[i]=L; lmin=std::min(lmin,L); lmax=std::max(lmax,L); } double range=std::max(1e-6,lmax-lmin); for(auto&v:lum) v=(v-lmin)/range; std::vector<double> sorted=lum; std::nth_element(sorted.begin(),sorted.begin()+sorted.size()*3/4,sorted.end()); double thr=sorted[sorted.size()*3/4]; double wx=0,wy=0,ws=0; for(unsigned y=0;y<obs.h;y++) for(unsigned x=0;x<obs.w;x++){ double w=std::max(0.0,lum[(size_t)y*obs.w+x]-thr); if(w<=0) continue; wx+=w*x; wy+=w*y; ws+=w; } if(ws<=1e-9){ cx=0.5*(obs.w-1); cy=0.5*(obs.h-1); } else { cx=wx/ws; cy=wy/ws; } int rmax=(int)std::ceil(std::sqrt((double)obs.w*obs.w+(double)obs.h*obs.h)); std::vector<double> radial((size_t)rmax+1,0.0), radial_w((size_t)rmax+1,0.0); for(unsigned y=0;y<obs.h;y++) for(unsigned x=0;x<obs.w;x++){ double dx=x-cx,dy=y-cy; int ri=(int)std::lround(std::sqrt(dx*dx+dy*dy)); if(ri<0||ri>rmax) continue; double w=std::max(0.0,lum[(size_t)y*obs.w+x]-thr*0.5); radial[(size_t)ri]+=w; radial_w[(size_t)ri]+=1.0; } for(size_t i=0;i<radial.size();++i) if(radial_w[i]>0) radial[i]/=radial_w[i]; int best_r=1; for(int i=2;i<rmax;i++) if(radial[(size_t)i]>radial[(size_t)best_r]) best_r=i; r_peak=(double)best_r; double var=0,sumw=0; for(int i=1;i<rmax;i++){ double w=std::max(0.0,radial[(size_t)i]); double d=i-r_peak; var+=w*d*d; sumw+=w; } r_sigma=std::sqrt(var/std::max(1e-6,sumw)); r_sigma=std::clamp(r_sigma,2.0,0.25*r_peak); double inner=std::max(1.0,r_peak-2.5*r_sigma), outer=std::min((double)rmax,r_peak+2.5*r_sigma); double width=std::max(1.0,outer-inner); std::vector<double> sr((size_t)tx*ty,0),sg((size_t)tx*ty,0),sb((size_t)tx*ty,0),sl((size_t)tx*ty,0),cnt((size_t)tx*ty,0); for(unsigned y=0;y<obs.h;y++) for(unsigned x=0;x<obs.w;x++){ double dx=x-cx,dy=y-cy; double rr=std::sqrt(dx*dx+dy*dy); if(rr<inner||rr>outer) continue; double t=(std::atan2(dy,dx)+M_PI)/(2.0*M_PI); if(t<0) t+=1.0; if(t>=1) t-=1.0; int c=std::clamp((int)std::floor(t*tx),0,tx-1); double rv=(rr-inner)/width; int r=std::clamp((int)std::floor(rv*ty),0,ty-1); size_t idx=(size_t)r*tx+c; const uint8_t* p=&obs.rgba[4ull*(y*obs.w+x)]; double R=p[0]/255.0,G=p[1]/255.0,B=p[2]/255.0,L=0.2126*R+0.7152*G+0.0722*B; sr[idx]+=R; sg[idx]+=G; sb[idx]+=B; sl[idx]+=L; cnt[idx]+=1.0; } tiles.assign((size_t)tx*ty,TileStat{}); for(int r=0;r<ty;r++) for(int c=0;c<tx;c++){ size_t idx=(size_t)r*tx+c; double n=std::max(1.0,cnt[idx]); double detail_boost=1.0+0.35*std::min(1.0,std::fabs(sl[idx]/n-0.5)); tiles[idx]=TileStat{std::clamp(detail_boost*sr[idx]/n,0.0,1.0),std::clamp(detail_boost*sg[idx]/n,0.0,1.0),std::clamp(detail_boost*sb[idx]/n,0.0,1.0),std::clamp(detail_boost*sl[idx]/n,0.0,1.0)}; } return true; }
bool generate_payload_dataset(const std::string& source_ppm,int tile_px_x,int tile_px_y,int ring_N,double ring_radius,double ring_sigma,const std::string& out_dir,std::string& dataset_csv,std::string& ring_preview_path,unsigned& srcW,unsigned& srcH,std::string& err){ ImageRGBA src; if(!read_ppm(source_ppm,src,err)) return false; srcW=src.w; srcH=src.h; int tx=0,ty=0; auto tiles=compute_tiles(src,tile_px_x,tile_px_y,tx,ty); fs::create_directories(out_dir); dataset_csv=tiles_to_csv(tiles,tx,ty); auto ring=render_ring_from_tiles(tiles,tx,ty,ring_N,ring_radius,ring_sigma); ring_preview_path=(fs::path(out_dir)/"ring_preview.ppm").string(); write_ppm(ring_preview_path,ring); ImageRGBA grid=src; draw_grid_overlay(grid,tile_px_x,tile_px_y); write_ppm((fs::path(out_dir)/"source_grid.ppm").string(),grid); return true; }
bool generate_payload_dataset_from_ring_observation(const std::string& ring_observation_ppm,int tile_px_x,int tile_px_y,const std::string& out_dir,std::string& dataset_csv,std::string& ring_preview_path,unsigned& srcW,unsigned& srcH,std::string& err){ ImageRGBA obs; if(!read_ppm(ring_observation_ppm,obs,err)) return false; srcW=obs.w; srcH=obs.h; int tx=std::clamp((int)obs.w/std::max(1,tile_px_x/2),16,256), ty=std::clamp((int)obs.h/std::max(1,tile_px_y/2),8,128); std::vector<TileStat> tiles; double cx=0,cy=0,rp=0,rs=0; if(!extract_tiles_from_ring_observation(obs,tx,ty,tiles,cx,cy,rp,rs)){ err="failed to extract ring tiles"; return false; } fs::create_directories(out_dir); dataset_csv=tiles_to_csv(tiles,tx,ty); ImageRGBA unwrapped=tiles_to_image(tiles,tx,ty); ring_preview_path=(fs::path(out_dir)/"ring_unwrapped_preview.ppm").string(); write_ppm(ring_preview_path,unwrapped); ImageRGBA marked=obs; int r0=(int)std::lround(rp), r1=(int)std::lround(rp-2.5*rs), r2=(int)std::lround(rp+2.5*rs); for(unsigned y=0;y<marked.h;y++) for(unsigned x=0;x<marked.w;x++){ double d=std::sqrt((x-cx)*(x-cx)+(y-cy)*(y-cy)); int di=(int)std::lround(d); if(di==r0||di==r1||di==r2){ size_t i=4ull*(y*marked.w+x); marked.rgba[i]=255; marked.rgba[i+1]=64; marked.rgba[i+2]=64; } } write_ppm((fs::path(out_dir)/"ring_detect_overlay.ppm").string(),marked); return true; }

namespace {
static inline double lum_at(const uint8_t* p) { return (0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]) / 255.0; }
static inline double clamp01d(double v) { return std::clamp(v, 0.0, 1.0); }

static ImageRGBA resize_bilinear(const ImageRGBA& in, unsigned out_w, unsigned out_h, uint8_t bg = 0) {
  ImageRGBA out;
  out.w = out_w;
  out.h = out_h;
  out.rgba.assign(4ull * out_w * out_h, 255);
  for (size_t i = 0; i < static_cast<size_t>(out_w) * out_h; ++i) {
    out.rgba[4 * i + 0] = bg;
    out.rgba[4 * i + 1] = bg;
    out.rgba[4 * i + 2] = bg;
  }
  if (in.w == 0 || in.h == 0) return out;
  const double sx = (in.w > 1 && out_w > 1) ? static_cast<double>(in.w - 1) / static_cast<double>(out_w - 1) : 0.0;
  const double sy = (in.h > 1 && out_h > 1) ? static_cast<double>(in.h - 1) / static_cast<double>(out_h - 1) : 0.0;
  for (unsigned y = 0; y < out_h; ++y) {
    const double fy = sy * y;
    const int y0 = static_cast<int>(std::floor(fy));
    const int y1 = std::min(static_cast<int>(in.h - 1), y0 + 1);
    const double wy = fy - y0;
    for (unsigned x = 0; x < out_w; ++x) {
      const double fx = sx * x;
      const int x0 = static_cast<int>(std::floor(fx));
      const int x1 = std::min(static_cast<int>(in.w - 1), x0 + 1);
      const double wx = fx - x0;
      const size_t i00 = 4ull * (static_cast<size_t>(y0) * in.w + static_cast<unsigned>(x0));
      const size_t i10 = 4ull * (static_cast<size_t>(y0) * in.w + static_cast<unsigned>(x1));
      const size_t i01 = 4ull * (static_cast<size_t>(y1) * in.w + static_cast<unsigned>(x0));
      const size_t i11 = 4ull * (static_cast<size_t>(y1) * in.w + static_cast<unsigned>(x1));
      const size_t di = 4ull * (static_cast<size_t>(y) * out_w + x);
      for (int c = 0; c < 3; ++c) {
        const double v00 = in.rgba[i00 + c];
        const double v10 = in.rgba[i10 + c];
        const double v01 = in.rgba[i01 + c];
        const double v11 = in.rgba[i11 + c];
        const double v0 = v00 + (v10 - v00) * wx;
        const double v1 = v01 + (v11 - v01) * wx;
        out.rgba[di + c] = static_cast<uint8_t>(std::lround(std::clamp(v0 + (v1 - v0) * wy, 0.0, 255.0)));
      }
      out.rgba[di + 3] = 255;
    }
  }
  return out;
}

struct BackgroundModel {
  double r = 0.0, g = 0.0, b = 0.0, l = 0.0, sat = 0.0;
};

struct PlanetCandidate {
  bool ok = false;
  int x0 = 0, y0 = 0, x1 = -1, y1 = -1;
  double cx = 0.0, cy = 0.0, radius = 0.0;
  double compactness = 0.0;
  double mask_coverage = 0.0;
  bool touches_edge = false;
  bool full_frame_like = false;
  int edge_touch_count = 0;
  std::vector<uint8_t> mask;
};

static double saturation_of(const uint8_t* p) {
  const double r = p[0] / 255.0, g = p[1] / 255.0, b = p[2] / 255.0;
  return std::max({r, g, b}) - std::min({r, g, b});
}

static BackgroundModel estimate_border_background(const ImageRGBA& in) {
  BackgroundModel bg{};
  if (in.w == 0 || in.h == 0) return bg;
  const unsigned border = std::max(2u, std::min(in.w, in.h) / 24u);
  double n = 0.0;
  for (unsigned y = 0; y < in.h; ++y) {
    for (unsigned x = 0; x < in.w; ++x) {
      if (x >= border && y >= border && x + border < in.w && y + border < in.h) continue;
      const uint8_t* p = &in.rgba[4ull * (static_cast<size_t>(y) * in.w + x)];
      bg.r += p[0] / 255.0;
      bg.g += p[1] / 255.0;
      bg.b += p[2] / 255.0;
      bg.l += lum_at(p);
      bg.sat += saturation_of(p);
      n += 1.0;
    }
  }
  if (n > 0.0) {
    bg.r /= n; bg.g /= n; bg.b /= n; bg.l /= n; bg.sat /= n;
  }
  return bg;
}

static bool detect_planet_candidate(const ImageRGBA& in, PlanetCandidate& out) {
  out = PlanetCandidate{};
  if (in.w == 0 || in.h == 0 || in.rgba.empty()) return false;
  const BackgroundModel bg = estimate_border_background(in);
  const size_t pix = static_cast<size_t>(in.w) * in.h;
  std::vector<uint8_t> candidate(pix, 0u);

  for (unsigned y = 0; y < in.h; ++y) {
    for (unsigned x = 0; x < in.w; ++x) {
      const size_t idx = static_cast<size_t>(y) * in.w + x;
      const uint8_t* p = &in.rgba[4ull * idx];
      const double r = p[0] / 255.0, g = p[1] / 255.0, b = p[2] / 255.0;
      const double l = lum_at(p);
      const double sat = saturation_of(p);
      const double color_d = std::sqrt((r - bg.r) * (r - bg.r) + (g - bg.g) * (g - bg.g) + (b - bg.b) * (b - bg.b));
      double edge = 0.0;
      if (x + 1 < in.w) edge = std::max(edge, std::fabs(lum_at(&in.rgba[4ull * (idx + 1)]) - l));
      if (y + 1 < in.h) edge = std::max(edge, std::fabs(lum_at(&in.rgba[4ull * (idx + in.w)]) - l));

      const bool chroma = (color_d > std::max(0.075, bg.sat + 0.04)) && sat > std::max(0.03, bg.sat + 0.015);
      const bool textured = edge > 0.05 && color_d > 0.055;
      const bool bright_detail = std::fabs(l - bg.l) > 0.10 && edge > 0.02;
      candidate[idx] = (chroma || textured || bright_detail) ? 1 : 0;
    }
  }

  std::vector<uint8_t> visited(pix, 0u);
  std::vector<size_t> stack;
  int best_x0 = 0, best_y0 = 0, best_x1 = -1, best_y1 = -1, best_touch_count = 0;
  size_t best_area = 0;
  double best_cx = 0.0, best_cy = 0.0, best_r = 0.0;
  double best_compact = 0.0;
  bool best_touches = false;
  bool best_full_frame = false;
  double best_score = -1.0;
  std::vector<uint8_t> best_mask(pix, 0u);

  for (unsigned sy = 0; sy < in.h; ++sy) {
    for (unsigned sx = 0; sx < in.w; ++sx) {
      const size_t start = static_cast<size_t>(sy) * in.w + sx;
      if (!candidate[start] || visited[start]) continue;
      int cx0 = static_cast<int>(sx), cy0 = static_cast<int>(sy), cx1 = static_cast<int>(sx), cy1 = static_cast<int>(sy);
      size_t area = 0;
      double sat_sum = 0.0, sum_x = 0.0, sum_y = 0.0;
      int edge_touch = 0;
      std::vector<size_t> pixels;
      stack.clear();
      stack.push_back(start);
      visited[start] = 1;
      while (!stack.empty()) {
        const size_t idx = stack.back();
        stack.pop_back();
        const int x = static_cast<int>(idx % in.w);
        const int y = static_cast<int>(idx / in.w);
        const uint8_t* p = &in.rgba[4ull * idx];
        area++;
        sat_sum += saturation_of(p);
        sum_x += x;
        sum_y += y;
        pixels.push_back(idx);
        cx0 = std::min(cx0, x); cy0 = std::min(cy0, y);
        cx1 = std::max(cx1, x); cy1 = std::max(cy1, y);
        if (x == 0) edge_touch |= 1;
        if (x == static_cast<int>(in.w) - 1) edge_touch |= 2;
        if (y == 0) edge_touch |= 4;
        if (y == static_cast<int>(in.h) - 1) edge_touch |= 8;
        const int nx[4] = {x - 1, x + 1, x, x};
        const int ny[4] = {y, y, y - 1, y + 1};
        for (int k = 0; k < 4; ++k) {
          if (nx[k] < 0 || ny[k] < 0 || nx[k] >= static_cast<int>(in.w) || ny[k] >= static_cast<int>(in.h)) continue;
          const size_t ni = static_cast<size_t>(ny[k]) * in.w + static_cast<unsigned>(nx[k]);
          if (!candidate[ni] || visited[ni]) continue;
          visited[ni] = 1;
          stack.push_back(ni);
        }
      }
      const int cw = cx1 - cx0 + 1;
      const int ch = cy1 - cy0 + 1;
      if (area < pix / 800 || cw < static_cast<int>(in.w / 12) || ch < static_cast<int>(in.h / 12)) continue;
      const double compact = static_cast<double>(area) / std::max(1, cw * ch);
      const int touch_count = ((edge_touch & 1) ? 1 : 0) + ((edge_touch & 2) ? 1 : 0) + ((edge_touch & 4) ? 1 : 0) + ((edge_touch & 8) ? 1 : 0);
      const bool full_frame_like = (cw > static_cast<int>(0.94 * in.w) && ch > static_cast<int>(0.94 * in.h)) || (compact > 0.80 && cw > static_cast<int>(0.90 * in.w) && ch > static_cast<int>(0.90 * in.h));
      const double cxm = sum_x / std::max<double>(1.0, area);
      const double cym = sum_y / std::max<double>(1.0, area);
      const double r = std::sqrt(static_cast<double>(area) / M_PI);
      double edge_penalty = (touch_count >= 2) ? 0.35 : ((touch_count == 1) ? 0.65 : 1.0);
      if (full_frame_like) edge_penalty *= 0.20;
      const double score = static_cast<double>(area) * (0.60 + 0.40 * compact) * edge_penalty * (1.0 + sat_sum / std::max<double>(1.0, area));
      if (score > best_score) {
        best_score = score;
        best_area = area;
        best_x0 = cx0; best_y0 = cy0; best_x1 = cx1; best_y1 = cy1;
        best_cx = cxm; best_cy = cym; best_r = r;
        best_compact = compact;
        best_touches = (touch_count > 0);
        best_touch_count = touch_count;
        best_full_frame = full_frame_like;
        std::fill(best_mask.begin(), best_mask.end(), 0u);
        for (size_t id : pixels) best_mask[id] = 1u;
      }
    }
  }
  if (best_area == 0 || best_x1 < best_x0 || best_y1 < best_y0) return false;
  out.ok = true;
  out.x0 = best_x0; out.y0 = best_y0; out.x1 = best_x1; out.y1 = best_y1;
  out.cx = best_cx; out.cy = best_cy; out.radius = best_r;
  out.compactness = best_compact;
  out.mask_coverage = static_cast<double>(best_area) / std::max<size_t>(1, pix);
  out.touches_edge = best_touches;
  out.edge_touch_count = best_touch_count;
  out.full_frame_like = best_full_frame;
  out.mask = std::move(best_mask);
  return true;
}

static std::array<double, 3> sample_rgb_bilinear(const ImageRGBA& img, double x, double y, uint8_t bg = 0) {
  if (img.w == 0 || img.h == 0) return {static_cast<double>(bg), static_cast<double>(bg), static_cast<double>(bg)};
  if (x < 0.0 || y < 0.0 || x > static_cast<double>(img.w - 1) || y > static_cast<double>(img.h - 1)) {
    return {static_cast<double>(bg), static_cast<double>(bg), static_cast<double>(bg)};
  }
  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int x1 = std::min(static_cast<int>(img.w - 1), x0 + 1);
  const int y1 = std::min(static_cast<int>(img.h - 1), y0 + 1);
  const double wx = x - x0;
  const double wy = y - y0;
  const size_t i00 = 4ull * (static_cast<size_t>(y0) * img.w + static_cast<unsigned>(x0));
  const size_t i10 = 4ull * (static_cast<size_t>(y0) * img.w + static_cast<unsigned>(x1));
  const size_t i01 = 4ull * (static_cast<size_t>(y1) * img.w + static_cast<unsigned>(x0));
  const size_t i11 = 4ull * (static_cast<size_t>(y1) * img.w + static_cast<unsigned>(x1));
  std::array<double, 3> out{};
  for (int c = 0; c < 3; ++c) {
    const double v00 = img.rgba[i00 + c], v10 = img.rgba[i10 + c], v01 = img.rgba[i01 + c], v11 = img.rgba[i11 + c];
    const double v0 = v00 + (v10 - v00) * wx;
    const double v1 = v01 + (v11 - v01) * wx;
    out[c] = std::clamp(v0 + (v1 - v0) * wy, 0.0, 255.0);
  }
  return out;
}
}  // namespace

bool precondition_source_image(const std::string& input_ppm,const std::string& out_dir,const SourcePreconditioningConfig& cfg,SourcePreconditioningResult& out,std::string& err) {
  out = SourcePreconditioningResult{};
  ImageRGBA in;
  if (!read_ppm(input_ppm, in, err)) return false;
  fs::create_directories(out_dir);
  const int bg = std::clamp(cfg.background_value, 0, 255);
  PlanetCandidate cand{};
  if (!detect_planet_candidate(in, cand)) {
    err = "failed to isolate planet candidate";
    return false;
  }
  const int bx0 = cand.x0, by0 = cand.y0, bx1 = cand.x1, by1 = cand.y1;
  const int bw = std::max(1, bx1 - bx0 + 1);
  const int bh = std::max(1, by1 - by0 + 1);
  const double ar = static_cast<double>(bw) / static_cast<double>(bh);
  std::string typ = cfg.object_type;
  if (typ == "auto") typ = (ar > 1.35) ? "ringed_planet" : "disk_planet";
  if (typ != "disk_planet" && typ != "ringed_planet" && typ != "extended_object") typ = "extended_object";
  double fill = (typ == "disk_planet") ? std::clamp(cfg.disk_fill_fraction, 0.58, 0.62) : std::clamp(cfg.extended_fill_fraction, 0.35, 0.95);
  const int N = std::max(128, cfg.canvas_N);
  const double pad_frac = std::clamp(cfg.object_padding_fraction, 0.0, 0.40);
  const double min_margin = std::clamp(cfg.minimum_source_margin_fraction, 0.01, 0.30);
  const double disk_r = std::max(4.0, cand.radius);
  const double margin_before_px = std::min({cand.cx - disk_r, cand.cy - disk_r, (in.w - 1) - (cand.cx + disk_r), (in.h - 1) - (cand.cy + disk_r)});
  out.source_margin_fraction_before = margin_before_px / std::max(1.0, static_cast<double>(std::min(in.w, in.h)));
  out.source_truncation_suspected =
      out.source_margin_fraction_before < min_margin ||
      cand.touches_edge ||
      cand.edge_touch_count >= 2 ||
      cand.full_frame_like ||
      cand.compactness < 0.30;
  out.detected_planet_center_x = cand.cx;
  out.detected_planet_center_y = cand.cy;
  out.detected_planet_radius_px = disk_r;
  out.mask_coverage_fraction = cand.mask_coverage;

  const double crop_half = disk_r * (1.0 + std::max(0.10, pad_frac));
  const int cx0 = std::max(0, static_cast<int>(std::floor(cand.cx - crop_half)));
  const int cy0 = std::max(0, static_cast<int>(std::floor(cand.cy - crop_half)));
  const int cx1 = std::min(static_cast<int>(in.w) - 1, static_cast<int>(std::ceil(cand.cx + crop_half)));
  const int cy1 = std::min(static_cast<int>(in.h) - 1, static_cast<int>(std::ceil(cand.cy + crop_half)));
  const int cw = std::max(1, cx1 - cx0 + 1);
  const int ch = std::max(1, cy1 - cy0 + 1);
  const int target_diameter = std::max(32, static_cast<int>(std::lround(fill * N)));
  const double s = static_cast<double>(target_diameter) / std::max(1.0, 2.0 * disk_r);
  const unsigned ow = static_cast<unsigned>(std::max(1, static_cast<int>(std::lround(cw * s))));
  const unsigned oh = static_cast<unsigned>(std::max(1, static_cast<int>(std::lround(ch * s))));
  if (ow + 2 >= static_cast<unsigned>(N) || oh + 2 >= static_cast<unsigned>(N)) {
    out.clipping_guard_triggered = true;
  }
  ImageRGBA canvas;
  canvas.w = static_cast<unsigned>(N);
  canvas.h = static_cast<unsigned>(N);
  canvas.rgba.assign(4ull * canvas.w * canvas.h, 255);
  for (size_t i = 0; i < static_cast<size_t>(N) * N; ++i) {
    canvas.rgba[4 * i + 0] = static_cast<uint8_t>(bg);
    canvas.rgba[4 * i + 1] = static_cast<uint8_t>(bg);
    canvas.rgba[4 * i + 2] = static_cast<uint8_t>(bg);
  }
  const int ox = std::max(0, (N - static_cast<int>(ow)) / 2);
  const int oy = std::max(0, (N - static_cast<int>(oh)) / 2);
  const double inv_s = (s > 0.0) ? 1.0 / s : 1.0;
  const double disk_cx = cand.cx;
  const double disk_cy = cand.cy;
  const double disk_rx = std::max(2.0, disk_r * 1.01);
  const double disk_ry = std::max(2.0, disk_r * 1.01);
  for (unsigned y = 0; y < oh; ++y) for (unsigned x = 0; x < ow; ++x) {
    const double sx = cx0 + (static_cast<double>(x) + 0.5) * inv_s;
    const double sy = cy0 + (static_cast<double>(y) + 0.5) * inv_s;
    bool keep = (typ != "disk_planet");
    if (typ == "disk_planet") {
      const double ex = (sx - disk_cx) / std::max(1.0, disk_rx);
      const double ey = (sy - disk_cy) / std::max(1.0, disk_ry);
      keep = (ex * ex + ey * ey) <= 1.0;
    }
    if (!keep) continue;
    auto rgb = sample_rgb_bilinear(in, sx, sy, static_cast<uint8_t>(bg));
    const size_t di = 4ull * (static_cast<size_t>(oy + static_cast<int>(y)) * canvas.w + static_cast<size_t>(ox + static_cast<int>(x)));
    canvas.rgba[di + 0] = static_cast<uint8_t>(std::lround(rgb[0]));
    canvas.rgba[di + 1] = static_cast<uint8_t>(std::lround(rgb[1]));
    canvas.rgba[di + 2] = static_cast<uint8_t>(std::lround(rgb[2]));
  }
  // Gentle deterministic normalization.
  double mean = 0.0;
  for (size_t i = 0; i < static_cast<size_t>(N) * N; ++i) mean += lum_at(&canvas.rgba[4 * i]);
  mean /= static_cast<double>(N) * N;
  const double tgt = 0.48;
  const double gain = std::clamp(tgt / std::max(0.08, mean), 0.80, 1.25);
  for (size_t i = 0; i < static_cast<size_t>(N) * N; ++i) for (int c = 0; c < 3; ++c) {
    canvas.rgba[4 * i + c] = static_cast<uint8_t>(std::lround(std::clamp(canvas.rgba[4 * i + c] * gain, 0.0, 255.0)));
  }
  ImageRGBA mask = canvas, overlay = canvas;
  ImageRGBA src_mask = in, src_overlay = in;
  for (size_t i = 0; i < static_cast<size_t>(in.w) * in.h; ++i) {
    const uint8_t v = (!cand.mask.empty() && cand.mask[i]) ? 255 : 0;
    src_mask.rgba[4 * i + 0] = v;
    src_mask.rgba[4 * i + 1] = v;
    src_mask.rgba[4 * i + 2] = v;
  }
  for (int x = bx0; x <= bx1; ++x) {
    for (int y : {by0, by1}) {
      const size_t di = 4ull * (static_cast<size_t>(y) * src_overlay.w + static_cast<size_t>(x));
      src_overlay.rgba[di + 0] = 255; src_overlay.rgba[di + 1] = 48; src_overlay.rgba[di + 2] = 48;
    }
  }
  for (int y = by0; y <= by1; ++y) {
    for (int x : {bx0, bx1}) {
      const size_t di = 4ull * (static_cast<size_t>(y) * src_overlay.w + static_cast<size_t>(x));
      src_overlay.rgba[di + 0] = 255; src_overlay.rgba[di + 1] = 48; src_overlay.rgba[di + 2] = 48;
    }
  }
  for (int x = cx0; x <= cx1; ++x) {
    for (int y : {cy0, cy1}) {
      const size_t di = 4ull * (static_cast<size_t>(y) * src_overlay.w + static_cast<size_t>(x));
      src_overlay.rgba[di + 0] = 255; src_overlay.rgba[di + 1] = 196; src_overlay.rgba[di + 2] = 0;
    }
  }
  for (int y = cy0; y <= cy1; ++y) {
    for (int x : {cx0, cx1}) {
      const size_t di = 4ull * (static_cast<size_t>(y) * src_overlay.w + static_cast<size_t>(x));
      src_overlay.rgba[di + 0] = 255; src_overlay.rgba[di + 1] = 196; src_overlay.rgba[di + 2] = 0;
    }
  }
  if (out.source_truncation_suspected) {
    for (unsigned x = 0; x < src_overlay.w; ++x) {
      for (unsigned y : {0u, src_overlay.h - 1u}) {
        const size_t di = 4ull * (static_cast<size_t>(y) * src_overlay.w + x);
        src_overlay.rgba[di + 0] = 255; src_overlay.rgba[di + 1] = 0; src_overlay.rgba[di + 2] = 0;
      }
    }
    for (unsigned y = 0; y < src_overlay.h; ++y) {
      for (unsigned x : {0u, src_overlay.w - 1u}) {
        const size_t di = 4ull * (static_cast<size_t>(y) * src_overlay.w + x);
        src_overlay.rgba[di + 0] = 255; src_overlay.rgba[di + 1] = 0; src_overlay.rgba[di + 2] = 0;
      }
    }
  }
  int mx0 = N, my0 = N, mx1 = -1, my1 = -1;
  for (size_t i = 0; i < static_cast<size_t>(N) * N; ++i) {
    const double l = lum_at(&canvas.rgba[4 * i]);
    const bool fg = std::fabs(l - (bg / 255.0)) > 0.05;
    const uint8_t v = fg ? 255 : 0;
    mask.rgba[4 * i + 0] = v; mask.rgba[4 * i + 1] = v; mask.rgba[4 * i + 2] = v;
    if (!fg) continue;
    const int x = static_cast<int>(i % N);
    const int y = static_cast<int>(i / N);
    mx0 = std::min(mx0, x); my0 = std::min(my0, y);
    mx1 = std::max(mx1, x); my1 = std::max(my1, y);
    overlay.rgba[4 * i + 1] = static_cast<uint8_t>(std::min(255, overlay.rgba[4 * i + 1] + 24));
  }
  if (mx1 >= mx0 && my1 >= my0) {
    const int margin_px = std::min({mx0, my0, N - 1 - mx1, N - 1 - my1});
    out.margin_fraction = static_cast<double>(margin_px) / static_cast<double>(N);
    if (out.margin_fraction < min_margin) out.clipping_guard_triggered = true;
    for (int x = mx0; x <= mx1; ++x) {
      for (int y : {my0, my1}) {
        const size_t di = 4ull * (static_cast<size_t>(y) * canvas.w + static_cast<size_t>(x));
        overlay.rgba[di + 0] = 255; overlay.rgba[di + 1] = 32; overlay.rgba[di + 2] = 32;
      }
    }
    for (int y = my0; y <= my1; ++y) {
      for (int x : {mx0, mx1}) {
        const size_t di = 4ull * (static_cast<size_t>(y) * canvas.w + static_cast<size_t>(x));
        overlay.rgba[di + 0] = 255; overlay.rgba[di + 1] = 32; overlay.rgba[di + 2] = 32;
      }
    }
  }
  out.preconditioned_source_path = (fs::path(out_dir) / "preconditioned_source.ppm").string();
  out.source_mask_path = (fs::path(out_dir) / "source_mask.ppm").string();
  out.source_overlay_path = (fs::path(out_dir) / "source_overlay.ppm").string();
  write_ppm(out.preconditioned_source_path, canvas);
  write_ppm(out.source_mask_path, src_mask);
  write_ppm(out.source_overlay_path, src_overlay);
  out.ok = true;
  out.object_type_detected = typ;
  out.bbox_x0 = bx0; out.bbox_y0 = by0; out.bbox_x1 = bx1; out.bbox_y1 = by1;
  out.fill_fraction_used = fill;
  out.method = (typ == "disk_planet") ? "disk_candidate_center_radius_norm" : "bbox_center_scale_norm";
  return true;
}

bool generate_sgl_observation_dataset(const std::string& preconditioned_source_ppm,const std::string& out_dir,const SglObservationConfig& cfg,SglObservationSummary& summary,std::string& dataset_descriptor,unsigned& srcW,unsigned& srcH,std::string& err) {
  summary = SglObservationSummary{};
  ImageRGBA src;
  if (!read_ppm(preconditioned_source_ppm, src, err)) return false;
  srcW = src.w;
  srcH = src.h;
  if (srcW == 0 || srcH == 0) { err = "preconditioned source empty"; return false; }
  fs::create_directories(out_dir);
  const int N = std::max(256, cfg.ring_sensor_N);
  const int A = std::max(256, cfg.ring_angular_samples);
  const int R = std::max(8, cfg.ring_radial_samples);
  const int obs = std::max(1, cfg.observation_count);
  const int ring_radius_px = std::clamp(static_cast<int>(std::lround(cfg.ring_radius_fraction * N)), 8, N / 2 - 4);
  const int ring_width_px = std::clamp(cfg.ring_radial_width_px, 4, N / 2 - 2);
  const double cx = 0.5 * (src.w - 1);
  const double cy = 0.5 * (src.h - 1);
  const double src_rmax = 0.48 * std::min(src.w, src.h);

  const std::string annulus_bin = (fs::path(out_dir) / "annulus_unwrapped.bin").string();
  const std::string obs_csv = (fs::path(out_dir) / "observations.csv").string();
  std::ofstream b(annulus_bin, std::ios::binary);
  if (!b) { err = "failed to open annulus bin"; return false; }
  std::ofstream o(obs_csv);
  if (!o) { err = "failed to open observations csv"; return false; }
  o << "obs_index,dx,dy,phase\n";
  const uint32_t hdr[4] = {static_cast<uint32_t>(obs), static_cast<uint32_t>(A), static_cast<uint32_t>(R), 3u};
  b.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));

  summary.ring_sensor_N = N;
  summary.ring_radius_px = ring_radius_px;
  summary.ring_radial_width_px = ring_width_px;
  summary.ring_angular_samples = A;
  summary.ring_radial_samples = R;
  summary.ring_processing_mode = cfg.ring_processing_mode;
  summary.annulus_bin_path = annulus_bin;
  summary.observations_csv_path = obs_csv;
  summary.ring_preview_path = (fs::path(out_dir) / "ring_preview.ppm").string();
  summary.active_annulus_pixels = static_cast<long long>(A) * R;
  summary.active_pixel_fraction = static_cast<double>(summary.active_annulus_pixels) / static_cast<double>(N) / static_cast<double>(N);

  const double golden = 2.399963229728653;
  std::vector<uint8_t> packed(static_cast<size_t>(A) * R * 3);
  for (int k = 0; k < obs; ++k) {
    const double t = (obs > 1) ? static_cast<double>(k) / (obs - 1) : 0.0;
    const double sr = 0.035 * std::sqrt(t);
    const double sa = k * golden;
    const double dx = sr * std::cos(sa);
    const double dy = sr * std::sin(sa);
    const double phase = 0.12 * k;
    o << k << "," << dx << "," << dy << "," << phase << "\n";

    for (int a = 0; a < A; ++a) {
      const double theta = (2.0 * M_PI * (a + 0.5) / A) + phase;
      for (int r = 0; r < R; ++r) {
        const double rn = (r + 0.5) / R;
        const double rr = src_rmax * rn;
        const double sx = cx + dx * src.w + rr * std::cos(theta);
        const double sy = cy + dy * src.h + rr * std::sin(theta);
        auto rgb = sample_rgb_bilinear(src, sx, sy, 0);
        const size_t pi = (static_cast<size_t>(a) * R + static_cast<size_t>(r)) * 3ull;
        packed[pi + 0] = static_cast<uint8_t>(std::lround(rgb[0]));
        packed[pi + 1] = static_cast<uint8_t>(std::lround(rgb[1]));
        packed[pi + 2] = static_cast<uint8_t>(std::lround(rgb[2]));
      }
    }
    b.write(reinterpret_cast<const char*>(packed.data()), static_cast<std::streamsize>(packed.size()));

    const bool save_frame = cfg.store_all_full_ring_frames_debug || cfg.store_full_ring_frames || (k == 0) || (cfg.store_ring_preview_every > 0 && (k % cfg.store_ring_preview_every == 0));
    if (save_frame) {
      ImageRGBA rf;
      rf.w = static_cast<unsigned>(N);
      rf.h = static_cast<unsigned>(N);
      rf.rgba.assign(4ull * rf.w * rf.h, 255);
      for (size_t i = 0; i < static_cast<size_t>(N) * N; ++i) {
        rf.rgba[4 * i + 0] = 0;
        rf.rgba[4 * i + 1] = 0;
        rf.rgba[4 * i + 2] = 0;
      }
      const double rcx = 0.5 * (N - 1) + dx * 0.05 * N;
      const double rcy = 0.5 * (N - 1) + dy * 0.05 * N;
      for (int a = 0; a < A; ++a) {
        const double theta = 2.0 * M_PI * (a + 0.5) / A;
        for (int r = 0; r < R; ++r) {
          const double rw = ((r + 0.5) / R - 0.5) * ring_width_px;
          const double rr = ring_radius_px + rw;
          const int x = static_cast<int>(std::lround(rcx + rr * std::cos(theta)));
          const int y = static_cast<int>(std::lround(rcy + rr * std::sin(theta)));
          if (x < 0 || y < 0 || x >= N || y >= N) continue;
          const size_t pi = (static_cast<size_t>(a) * R + static_cast<size_t>(r)) * 3ull;
          const size_t di = 4ull * (static_cast<size_t>(y) * rf.w + static_cast<size_t>(x));
          rf.rgba[di + 0] = std::max(rf.rgba[di + 0], packed[pi + 0]);
          rf.rgba[di + 1] = std::max(rf.rgba[di + 1], packed[pi + 1]);
          rf.rgba[di + 2] = std::max(rf.rgba[di + 2], packed[pi + 2]);
        }
      }
      const std::string p = (k == 0) ? summary.ring_preview_path : (fs::path(out_dir) / ("ring_preview_" + std::to_string(k) + ".ppm")).string();
      write_ppm(p, rf);
    }
  }

  std::ostringstream ds;
  ds << "format,sgl_annulus_v2\n";
  ds << "dataset_dir," << out_dir << "\n";
  ds << "source_path," << preconditioned_source_ppm << "\n";
  ds << "source_w," << src.w << "\n";
  ds << "source_h," << src.h << "\n";
  ds << "observation_count," << obs << "\n";
  ds << "ring_sensor_N," << N << "\n";
  ds << "ring_radius_px," << ring_radius_px << "\n";
  ds << "ring_radial_width_px," << ring_width_px << "\n";
  ds << "ring_angular_samples," << A << "\n";
  ds << "ring_radial_samples," << R << "\n";
  ds << "ring_processing_mode," << cfg.ring_processing_mode << "\n";
  ds << "annulus_bin_path," << annulus_bin << "\n";
  ds << "observations_csv_path," << obs_csv << "\n";
  ds << "ring_preview_path," << summary.ring_preview_path << "\n";
  dataset_descriptor = ds.str();
  return true;
}

bool is_sgl_dataset_descriptor(const std::string& descriptor) {
  return descriptor.rfind("format,sgl_annulus_v2", 0) == 0;
}
} // namespace sgl
