#ifndef TREC_RUN_FILE_HPP
#define TREC_RUN_FILE_HPP

#include <string>
#include <vector>

namespace indri {
namespace query {
// TREC formatted output: queryNumber, Q0, documentName, rank, score, runID
class TrecRecord {
 public:
  std::string queryNumber;
  std::string q0;
  std::string documentName;
  int rank;
  double score;
  std::string runID;
};

class TrecRunFile {
 public:
  typedef std::vector<TrecRecord>::iterator iterator;
  std::vector<TrecRecord> load(const std::string &path);
  std::vector<TrecRecord> load(std::istream &input);
};
}
}

#endif
