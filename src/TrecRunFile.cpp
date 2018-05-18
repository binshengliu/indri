#include "indri/TrecRunFile.hpp"
#include <fstream>
#include <sstream>

// TREC formatted output: queryNumber, Q0, documentName, rank, score, runID
std::vector<indri::query::TrecRecord> indri::query::TrecRunFile::load(const std::string &path) {
  std::ifstream infile(path);
  return load(infile);
}

std::vector<indri::query::TrecRecord> indri::query::TrecRunFile::load(std::istream &input) {
  std::vector<indri::query::TrecRecord> records;
  TrecRecord r;
  while (input >> r.queryNumber >> r.q0 >> r.documentName
         >> r.rank >> r.score >> r.runID) {
    records.push_back(r);
  }
  return records;
}
