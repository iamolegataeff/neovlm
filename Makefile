CC = cc
CFLAGS = -O2 -DUSE_BLAS -DACCELERATE
LDFLAGS = -lm -framework Accelerate

neovlm: neovlm.c notorch.c notorch.h
	$(CC) $(CFLAGS) neovlm.c notorch.c $(LDFLAGS) -o neovlm

debug: neovlm.c notorch.c notorch.h
	$(CC) -g -fsanitize=address -DUSE_BLAS -DACCELERATE neovlm.c notorch.c $(LDFLAGS) -o neovlm

clean:
	rm -f neovlm

.PHONY: clean debug
