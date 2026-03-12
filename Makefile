all: \
	index.md \
	0_intro.ipynb \
	1_create_lease.ipynb \
	2_create_server_nvidia.ipynb \
	3_fastapi_setup.ipynb \
	workspace/4_fastapi.ipynb \
	workspace/5_triton.ipynb

clean: 
	rm index.md \
	0_intro.ipynb \
	1_create_lease.ipynb \
	2_create_server_nvidia.ipynb \
	3_fastapi_setup.ipynb \
	workspace/4_fastapi.ipynb \
	workspace/5_triton.ipynb \
	workspace/fastapi.ipynb \
	workspace/triton.ipynb

index.md: snippets/*.md 
	cat snippets/intro.md \
		snippets/create_lease.md \
		snippets/create_server_nvidia.md \
		snippets/fastapi_setup.md \
		snippets/fastapi.md \
		snippets/triton.md \
		> index.tmp.md
	grep -v '^:::' index.tmp.md > index.md
	rm index.tmp.md
	cat snippets/footer.md >> index.md

0_intro.ipynb: snippets/intro.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i snippets/frontmatter_python.md snippets/intro.md \
                -o 0_intro.ipynb  
	sed -i 's/attachment://g' 0_intro.ipynb

1_create_lease.ipynb: snippets/create_lease.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i snippets/frontmatter_python.md snippets/create_lease.md \
                -o 1_create_lease.ipynb  
	sed -i 's/attachment://g' 1_create_lease.ipynb


2_create_server_nvidia.ipynb: snippets/create_server_nvidia.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i snippets/frontmatter_python.md snippets/create_server_nvidia.md \
                -o 2_create_server_nvidia.ipynb  
	sed -i 's/attachment://g' 2_create_server_nvidia.ipynb

3_fastapi_setup.ipynb: snippets/fastapi_setup.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i snippets/frontmatter_python.md snippets/fastapi_setup.md \
                -o 3_fastapi_setup.ipynb  
	sed -i 's/attachment://g' 3_fastapi_setup.ipynb

workspace/4_fastapi.ipynb: snippets/fastapi.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/fastapi.md \
				-o workspace/4_fastapi.ipynb  
	sed -i 's/attachment://g' workspace/4_fastapi.ipynb

workspace/5_triton.ipynb : snippets/triton.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_bash.md snippets/triton.md \
				-o workspace/5_triton.ipynb  
	sed -i 's/attachment://g' workspace/5_triton.ipynb
