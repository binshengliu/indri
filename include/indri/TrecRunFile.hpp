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

class TrecQueryResult {
 public:
  std::string queryNumber;
  std::vector<TrecRecord> records;
};

class TrecRunFile {
 public:
  std::vector<TrecQueryResult> load(const std::string &path, size_t countPerQuery);
  std::vector<TrecQueryResult> load(std::istream &input, size_t countPerQuery);
};
}
}

#endif
