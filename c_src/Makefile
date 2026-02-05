# viva_tensor NIF Makefile - Apple Accelerate on macOS
#
# Usage:
#   make          - Build the NIF
#   make clean    - Remove build artifacts
#   make test     - Verify NIF builds correctly

# Detect OS - only macOS supported
UNAME_S := $(shell uname -s)

ifneq ($(UNAME_S),Darwin)
$(error This NIF requires macOS with Apple Accelerate framework)
endif

# Compiler
CC = clang

# Detect architecture
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
    ARCH_FLAGS = -march=native -mavx2
else ifeq ($(UNAME_M),arm64)
    ARCH_FLAGS = -march=native
else
    ARCH_FLAGS = -march=native
endif

# Erlang paths
ERL_ROOT := $(shell erl -noshell -eval 'io:format("~s", [code:root_dir()]).' -s init stop)
ERL_INCLUDE := $(wildcard $(ERL_ROOT)/erts-*/include)

# Compiler flags
CFLAGS = -O3 -fPIC -Wall -Wextra -Wno-unused-parameter
CFLAGS += -framework Accelerate
CFLAGS += $(ARCH_FLAGS)
CFLAGS += -I$(ERL_INCLUDE)

# Linker flags for shared library
LDFLAGS = -dynamiclib -undefined dynamic_lookup

# Target
TARGET = ../priv/viva_tensor_nif.so

# Sources
SRCS = viva_nif.c
OBJS = $(SRCS:.c=.o)

# Build rules
.PHONY: all clean test info

all: $(TARGET)

$(TARGET): $(OBJS) | ../priv
	$(CC) $(LDFLAGS) -framework Accelerate -o $@ $^
	@echo "Built: $@"

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

../priv:
	mkdir -p ../priv

clean:
	rm -f $(OBJS) $(TARGET)

test: $(TARGET)
	@echo "NIF built successfully at $(TARGET)"
	@file $(TARGET)

info:
	@echo "CC:          $(CC)"
	@echo "ARCH:        $(UNAME_M)"
	@echo "ERL_ROOT:    $(ERL_ROOT)"
	@echo "ERL_INCLUDE: $(ERL_INCLUDE)"
	@echo "CFLAGS:      $(CFLAGS)"
