NUM_JOBS = 8
CXX      = g++


CMAKE_CMD = mkdir -p build && cd build && cmake ..

FLAGS =
DEBUG_FLAGS = $(FLAGS) -DCMAKE_BUILD_TYPE:STRING=Debug
RELEASE_FLAGS = $(FLAGS) -DCMAKE_BUILD_TYPE:STRING=Release


all :
	@$(CMAKE_CMD) $(DEBUG_FLAGS) && make -s -j$(NUM_JOBS)

run :
	@cd build && cp ../tests/test_flash_mla_sm80.py . && cp ../tests/test_flash_mla_sm86.py . && cp ../tests/test_flash_mla_sm89.py . && cp ../tests/test_flash_mla_sm90.py . && cp -r ../flash_mla .  && python ./test_flash_mla_sm80.py

run_sm86 :
	@cd build && cp ../tests/test_flash_mla_sm86.py . && cp -r ../flash_mla .  && python ./test_flash_mla_sm86.py

run_sm89 :
	@cd build && cp ../tests/test_flash_mla_sm89.py . && cp -r ../flash_mla .  && python ./test_flash_mla_sm89.py


clean:
	read -r -p "This will delete the contents of build/*. Are you sure? [CTRL-C to abort] " response && rm -rf build/*


.PHONY: all clean run run_sm86 run_sm89

