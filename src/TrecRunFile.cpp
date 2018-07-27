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
  std::string score;
  while (input >> r.queryNumber >> r.q0 >> r.documentName
         >> r.rank >> score >> r.runID) {
    if (q.queryNumber.empty()) {
      q.queryNumber = r.queryNumber;
    }

    if (score == "-inf") {
      continue;
    }
    r.score = std::stod(score);

    if (r.queryNumber == q.queryNumber && q.records.size() >= countPerQuery) {
      continue;
    } else if (r.queryNumber != q.queryNumber) {
      if (q.records.size() < countPerQuery) {
        results.push_back(q);
      }
      q.queryNumber = r.queryNumber;
      q.records.clear();
    }
    q.records.push_back(r);
    if (q.records.size() == countPerQuery) {
      results.push_back(q);
    }
  }

  if (q.records.size() > 0 && q.records.size() < countPerQuery) {
    results.push_back(q);
  }

  return results;
}
