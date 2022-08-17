import click

import subprocess

@click.command()
@click.argument('fresnel', type=click.Path(exists=True))
def main(fresnel):

    # clear build directory
    cmd = f"rm -r .build/fresnel"
    subprocess.run(cmd.split())

    # configure cmake project
    cmd = f"cmake -B .build/fresnel -S {fresnel} -GNinja -DCMAKE_CXX_FLAGS=-march=native -DCMAKE_C_FLAGS=-march=native -DENABLE_OPTIX=ON -DENABLE_EMBREE=OFF"
    subprocess.run(cmd.split())

    cmd = "cmake --build .build/fresnel -j8"
    subprocess.run(cmd.split())

    cmd = "cmake --install .build/fresnel"
    subprocess.run(cmd.split())

if __name__ == '__main__':
    main()