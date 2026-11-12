/* ds4-router-train: ReMoE router fine-tuning CLI (Part 1).
 * Thin main over ds4_router_train_main in ds4.c. */
#include "ds4.h"

int main(int argc, char **argv) {
    return ds4_router_train_main(argc, argv);
}
