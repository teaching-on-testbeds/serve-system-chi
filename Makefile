all: \
	index.md \
	0_intro.ipynb \
	1_create_server_nvidia.ipynb \
	workspace/fastapi.ipynb \
	workspace/triton.ipynb

clean: 
	rm index.md \
	0_intro.ipynb \
	1_create_server_nvidia.ipynb \
	workspace/fastapi.ipynb \
	workspace/triton.ipynb

index.md: snippets/*.md 
	cat snippets/intro.md \
		snippets/create_server_nvidia.md \
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


1_create_server_nvidia.ipynb: snippets/create_server_nvidia.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
                -i snippets/frontmatter_python.md snippets/create_server_nvidia.md \
                -o 1_create_server_nvidia.ipynb  
	sed -i 's/attachment://g' 1_create_server_nvidia.ipynb

workspace/fastapi.ipynb: snippets/fastapi.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_python.md snippets/fastapi.md \
				-o workspace/fastapi.ipynb  
	sed -i 's/attachment://g' workspace/fastapi.ipynb

workspace/triton.ipynb : snippets/triton.md
	pandoc --resource-path=../ --embed-resources --standalone --wrap=none \
				-i snippets/frontmatter_bash.md snippets/triton.md \
				-o workspace/triton.ipynb  
	sed -i 's/attachment://g' workspace/triton.ipynb