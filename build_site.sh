echo "converting notebooks..."

jupyter nbconvert --to html 'index.ipynb' --template='full.tpl'
jupyter nbconvert --to html '1 Introduction.ipynb' --template=config/web.tpl
jupyter nbconvert --to html '2.1 Yanking.ipynb' --template=config/web.tpl
jupyter nbconvert --to html '2.2 Teleportation.ipynb' --template=config/web.tpl
jupyter nbconvert --to html '2.3 Time Evolution.ipynb' --template=config/web.tpl
jupyter nbconvert --to html '2.4 The TFD.ipynb' --template=config/web.tpl
jupyter nbconvert --to html '2.5 Scrambling and Winding.ipynb' --template=config/web.tpl
jupyter nbconvert --to html '2.6 Wormhole Experiments.ipynb' --template=config/web.tpl
jupyter nbconvert --to html '2.7 Geometrical Sketch.ipynb' --template=config/web.tpl
jupyter nbconvert --to html '3.1 Philosophical Interlude.ipynb' --template=config/web.tpl
jupyter nbconvert --to html '3.2 Majorana Fermions and SYK.ipynb' --template=config/web.tpl
jupyter nbconvert --to html '3.3 A Really Existing Quantum Computer.ipynb' --template=config/web.tpl
jupyter nbconvert --to html '4.1 Spheres.ipynb' --template=config/web.tpl
jupyter nbconvert --to html '4.2 Quantum Clocks.ipynb' --template=config/web.tpl
jupyter nbconvert --to html '4.3 Representations.ipynb' --template=config/web.tpl
jupyter nbconvert --to html '5.1 Atoms 0.ipynb' --template=config/web.tpl
jupyter nbconvert --to html '5.2 Atoms 1.ipynb' --template=config/web.tpl
jupyter nbconvert --to html '5.3 Atoms 2.ipynb' --template=config/web.tpl
jupyter nbconvert --to html '5.4 Atoms 3.ipynb' --template=config/web.tpl
jupyter nbconvert --to html '5.5 Atoms 4.ipynb' --template=config/web.tpl
jupyter nbconvert --to html '5.6 Atoms 5.ipynb' --template=config/web.tpl
jupyter nbconvert --to html '5.7 Atoms 6.ipynb' --template=config/web.tpl
jupyter nbconvert --to html '5.8 Atoms 7.ipynb' --template=config/web.tpl

echo "uploading..."

git add *
git commit -m "!"
git push origin master