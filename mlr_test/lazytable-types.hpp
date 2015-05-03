#ifndef LAZYTABLE_TYPES_HPP_
#define LAZYTABLE_TYPES_HPP_

#include <stdint.h>

#define ROW_DATA_SIZE 21504
typedef float val_t;
struct ArrayData {
  val_t data[ROW_DATA_SIZE];
  void init() {
    for (uint32_t i = 0; i < ROW_DATA_SIZE; i++) {
      data[i] = 0;
    }
  }
  ArrayData() {
    init();
  }
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & data;
  }
};
typedef ArrayData RowData;
typedef ArrayData RowOpVal;

#endif  /* LAZYTABLE_TYPES_HPP_ */