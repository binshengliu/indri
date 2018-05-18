#include "indri/TrecRunFile.hpp"
#include <fstream>
#include <sstream>

// TREC formatted output: queryNumber, Q0, documentName, rank, score, runID
std::vector<indri::query::TrecQueryResult> indri::query::TrecRunFile::load(const std::string &path, size_t countPerQuery) {
  std::ifstream infile(path);
  return load(infile, countPerQuery);
}

std::vector<indri::query::TrecQueryResult> indri::query::TrecRunFile::load(std::istream &input, size_t countPerQuery) {
  std::vector<indri::query::TrecQueryResult> results;

  TrecQueryResult q;
  TrecRecord r;
  while (input >> r.queryNumber >> r.q0 >> r.documentName
         >> r.rank >> r.score >> r.runID) {
    if (r.queryNumber == q.queryNumber && q.records.size() >= countPerQuery) {
      continue;
    } else if (r.queryNumber != q.queryNumber) {
      q.queryNumber = r.queryNumber;
      q.records.clear();
    }
    q.records.push_back(r);
    if (q.records.size() == countPerQuery) {
      results.push_back(q);
    }
  }
  return results;
}
