# Developer Instructions

## Documentation Generation

To generate and view a local version of the documentation, run

```shell
cd docs
julia make.jl local
```

then open "docs/build/index.html" in your browser.

## Code Reloading

When changing the code of BAT.jl and testing snippets and examples in the REPL, automatic code reloading comes in very handy. Try out [Revise.jl](https://github.com/timholy/Revise.jl).
