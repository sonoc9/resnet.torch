--
-- S
end

function ResidualDrop:updateGradInput(input, gradOutput)
   self.gradInput = self.gradInput or input.new()
   self.gradInput:resizeAs(input):copy(self.skip:updateGradInput(input, gradOutput))
   if self.gate then
      self.gradInput:add(self.net:updateGradInput(input, gradOutput))
   end
   return self.gradInput
end

function ResidualDrop:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if self.gate then
      self.net:accGradParameters(input, gradOutput, scale)
   end
end

-- END of class

--
-- Adds a residual block to the passed in model
--
     end
   else
     for i=1, count do
       addResidualDrop(model, deathRate, nOutChannels, nOutChannels, 1)
     end
   end
end

local function createModel(opt)
  -- Saves 40% time according to http://torch.ch/blog/2016/02/04/resnets.html
  local cfg = {
     [18]  = {{2, 2, 2, 2}, 512},
     [34]  = {{3, 4, 6, 3}, 512},
     [50]  = {{3, 4, 6, 3}, 2048},
     [101] = {{3, 4, 23, 3}, 2048},
     [152] = {{3, 8, 36, 3}, 2048},
  }
  local depth = opt.depth
  assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
  local def, nFeatures = table.unpack(cfg[depth])

  iChannels = 64
  print(' | ResNet-' .. depth .. ' ImageNet')

  ---- Buidling the residual network model ----
  -- Input: 3x32x32
  print('Building model...')
  model = nn.Sequential()
  ------> 3, 32,32
  model:add(Convolution(3,64,7,7,2,2,3,3)) -- 64
  model:add(SBatchNorm(64))
  model:add(ReLU(3,8))
  model:add(Max(3,3,2,2,1,1)) -- 32
  layer(model, nil, 64, 64, def[1], 1) -- 32
  layer(model, nil, 64, 128, def[2], 2) -- 16
  layer(model, nil, 128
        end
     end
  end
  local function BNInit(name)
     for k,v in pairs(model:findModules(name)) do
        v.weight:fill(1)
        v.bias:zero()
     end
  end
  ConvInit('cudnn.SpatialConvolution')
  ConvInit('nn.SpatialConvolution')
  BNInit('fbnn.SpatialBatchNormalization')
  BNInit('SBatchNorm')
  BNInit('nn.SpatialBatchNormalization')
  for k,v in pairs(model:findModules('nn.Linear')) do
     v.bias:zero()
  end
  model:cuda()

  if opt.cudnn == 'deterministic' then
     model:apply(function(m)
        if m.setMode then m:setMode(1,1,1) end
     end)
  end

  model:get(1).gradInput = nil

  return model
end

return createModel
