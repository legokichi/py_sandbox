import numpy as np
import sys
import time
from PIL import Image
import OpenGL
OpenGL.ERROR_ON_COPY = True
# http://pyopengl.sourceforge.net/documentation/pydoc/OpenGL.arrays.arrayhelpers.html
# setInputArraySizeType( baseOperation , size , type , argName = 0 )
#  Decorate function with vector-handling code for a single argument
#  if OpenGL.ERROR_ON_COPY is False, then we return the named argument, converting to the passed array type, optionally checking that the array matches size.
#  if OpenGL.ERROR_ON_COPY is True, then we will dramatically simplify this function, only wrapping if size is True, i.e. only wrapping if we intend to do a size check on the array.
OpenGL.ERROR_CHECKING = False
# http://pyopengl.sourceforge.net/documentation/opengl_diffs.html
#   You can disable PyOpenGL's error checking by setting a module-level flag in the OpenGL package before importing any of the sub-modules, like so:
#   This will tend to cause a huge speed increase in your code, as the number of OpenGL calls issued will roughly halve compared to the error-checking version of the same script. 
import OpenGL.GL as GL
import OpenGL.GLU as GLU # The OpenGL Utility Library - https://en.wikipedia.org/wiki/OpenGL_Utility_Library
import OpenGL.GLUT as GLUT # the OpenGL Utilitiy Toolkit - https://ja.wikipedia.org/wiki/OpenGL_Utility_Toolkit

# PyOpenGL 3.0.1 introduces this convenience module...
import OpenGL.GL.shaders as shaders

vertices = None
indices = None

def InitGL( vertex_shade_code, fragment_shader_code, texture_image ):
    glClearColor(0.0, 0.0, 0.0, 0.0)

    texture_id = glGenTextures( 1 )
    glPixelStorei( GL_UNPACK_ALIGNMENT, 1 )
    glActiveTexture( GL_TEXTURE0 )
    glBindTexture( GL_TEXTURE_2D, texture_id )

    if texture_image.mode == 'RGB':
        glTexImage2D( GL_TEXTURE_2D,
                      0,
                      4,
                      texture_image.size[0],
                      texture_image.size[1],
                      0,
                      GL_RGB,
                      GL_UNSIGNED_BYTE,
                      texture_image.tobytes() )
    else:
        glTexImage2D( GL_TEXTURE_2D,
                      0,
                      4,
                      texture_image.size[0],
                      texture_image.size[1],
                      0,
                      GL_RGBA,
                      GL_UNSIGNED_BYTE,
                      texture_image.tobytes() )
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR )
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR )
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE )
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE )

    program = compileProgram(
        compileShader( vertex_shade_code,GL_VERTEX_SHADER),
        compileShader( fragment_shader_code,GL_FRAGMENT_SHADER),)

    glUseProgram(program)
    glUniform1i( glGetUniformLocation( program, "s_texture" ), 0 );
    glUniform1f( glGetUniformLocation( program, "texture_width" ), float( texture_image.size[ 0 ] ) )
    glUniform1f( glGetUniformLocation( program, "texture_height" ), float( texture_image.size[ 1 ] ) )


    global vertices
    global indices
    position_vertices = [ -1.0,  1.0, 0.0,
                          -1.0, -1.0, 0.0,
                           1.0, -1.0, 0.0,
                           1.0,  1.0, 0.0, ]
    texture_vertices = [ 0.0, 0.0,
                         0.0, texture_image.size[ 1 ],
                         texture_image.size[ 0 ], texture_image.size[ 1 ],
                         texture_image.size[ 0 ], 0.0 ]

    indices = [ 0, 1, 2, 0, 2, 3 ]

    position_loc = glGetAttribLocation( program, 'a_position' )
    glVertexAttribPointer( position_loc,
                           3,
                           GL_FLOAT,
                           GL_FALSE,
                           3 * 4,
                           np.array( position_vertices, np.float32 ) )

    tex_loc = glGetAttribLocation( program, 'a_texCoord' )
    glVertexAttribPointer( tex_loc,
                           2,
                           GL_FLOAT,
                           GL_FALSE,
                           2 * 4,
                           np.array( texture_vertices, np.float32 ) )

    glEnableVertexAttribArray( position_loc )
    glEnableVertexAttribArray( tex_loc )







def initGLUT(vertex_shade_code, fragment_shader_code, texture_image):
    GLUT.glutInit(sys.argv) # argv,argcを渡す
    # 表示モード
    if texture_image.mode == 'RGB':
        GLUT.glutInitDisplayMode(GLUT.GLUT_RGB | GLUT.GLUT_DOUBLE | GLUT.GLUT_DEPTH)
    elif texture_image.mode == 'RGBA':
        GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DOUBLE | GLUT.GLUT_DEPTH)
    else:
        # 謎のモード
        print(texture_image.mode)
        exit()
    window_width, window_height = texture_image.size # テクスチャサイズ = ウィンドウサイズ
    GLUT.glutInitWindowSize( window_width, window_height )
    # the window starts at the upper left corner of the screen
    GLUT.glutInitWindowPosition(0, 0)
    GLUT.glutCreateWindow( sys.argv[0] )

    ## The main drawing function.
    def draw():
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable( GL.GL_TEXTURE_2D )
        GL.glDrawElements( GL.GL_TRIANGLES, 6, GL.GL_UNSIGNED_SHORT, np.array( indices, np.uint16 ) )
        GL.glDisable(GL.GL_TEXTURE_2D)
        GLUT.glutSwapBuffers()

    GLUT.glutDisplayFunc(draw)
    GLUT.glutIdleFunc(drawGLScene) # When we are doing nothing, redraw the scene.
    
    initGL( vertex_shade_code, fragment_shader_code, texture_image ) # Initialize our window.

    GLUT.glutMainLoop() # Start Event Processing Engine

if __name__ == "__main__":
    vertex_shader_file = "shader-vert.c"
    fragment_shader_file = "shader-flag.c"
    texture_file = "2016-10-18-123734.jpg"

    # コード読み込み
    vertex_shade_code = '\n'.join( open( vertex_shader_file, 'r' ).readlines() )
    fragment_shader_code = '\n'.join( open( fragment_shader_file, 'r' ).readlines() )
    # 画像読み込み
    texture_image = Image.open( texture_file )
    print(type(texture_image), dir(texture_image))

    initGLUT(vertex_shade_code, fragment_shader_code, texture_image)


