
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <fstream>
#include <sstream>

#include "timing.h"

struct BlockWallTimeSorter
{
  BlockWallTimeSorter(std::map<std::string, TimingData>& blocks_)
    : blocks(blocks_)
  { } ;

  bool operator()(const std::string& key1,const std::string& key2) {
    return blocks[key1].ttot > blocks[key2].ttot ;
  }
private:
  std::map<std::string,TimingData>& blocks;
} ;


// Use a depth-first search to collect total timing data at each level
std::vector<std::map<std::string,TimingData>>
collect_timing_data(TimingNode* node,
		    std::vector<std::map<std::string,TimingData>> &&totals) {

  for (auto & child : node->children)
    totals = collect_timing_data(&child.second, std::move(totals)) ;

  // We have reached the top
  if (node->block_name == "") return std::move(totals) ;
  
  if (totals.size() <= (unsigned int) node->timing_level)
    totals.resize(node->timing_level + 1) ;

  totals[node->timing_level][node->block_name] += node->data;

  return std::move(totals) ;
}

void CodeTiming::SaveTimingStatistics() 
{
  using std::ios ;
  using std::setw;
  
  // Skip if output file not specified
  if (do_not_time) return ;

  std::string dashes(100,'-');

  double ttot = RunningTime() ;

  /* Don't bother if we've been going too little */
  if (ttot == 0) return ;
  
  std::ofstream outfile((basename + ".timing").c_str()) ;
  outfile << resetiosflags(ios::adjustfield);
  outfile << setiosflags(ios::left);
  outfile << dashes << "\n";
  outfile << "Total simulation wall clock time : " << ttot << "\n";  
  outfile << "\n" ;

  outfile << dashes << "\n";


  // Create the array of total times
  std::vector<std::map<std::string,TimingData>> totals ;

  totals = collect_timing_data(&root_node, std::move(totals)) ;
  
  // Output timing data on each hierarchical level
  for (unsigned int l=0; l<totals.size(); l++) {
    double tcount = 0.0;
    outfile << "Level : " << l << "\n";
    outfile << "Block";
    outfile << std::string(45, ' ' );
    outfile << "Total time" << std::string(5, ' ')
	    << "Percentage" << std::string(5, ' ')
	    << "Times called" 
	    << "\n" ;
    outfile << dashes << "\n";

    // First sort by wall time
    std::vector<std::string> keys ;
    for(auto & iter : totals[l])
      keys.push_back(iter.first) ;

    std::sort(keys.begin(), keys.end(), BlockWallTimeSorter(totals[l])) ;

    for(auto & key : keys) {
      auto block = totals[l][key] ;
      tcount += block.ttot;
      double tfraction = block.ttot / ttot;
      outfile << setw(50) << key
              << setw(15) << block.ttot
              << setw(15) << 100.0*tfraction
	      << setw(15) << block.Ncalled
              << "\n";
    }

    outfile << setw(50) << "REMAINDER" ;
    outfile << setw(15) << (ttot - tcount) 
            << setw(15) << 100.0*(ttot - tcount)/ttot
            << "\n";
    outfile << dashes << "\n";
  }
  outfile << resetiosflags(ios::adjustfield);
  outfile.close();

  return;
}

std::unique_ptr<CodeTiming> timer(new CodeTiming) ;
