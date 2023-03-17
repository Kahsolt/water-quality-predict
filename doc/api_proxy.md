# Proxy API documentation

=> For native API doc, see [/doc/api](/doc/api)

### General 总论

用于对接平台方要求的前端 API 和本项目的 API

----

### GET|POST /page2/getFittingCurve 查询【回归预测】回归模型拟合曲线

```typescript
// request
interface {
  dischargeCode: str    // 排口编码
  monitorFactor: str    // 因子编码
  start: str            // 查询周期开始时间 "2023-03-14 10:00:00"
  end: str              // 查询周期结束时间
}

// response
interface {

}
```

### GET|POST /page2/get6hPredictionResult 查询【回归预测】未来6h预测结果

```typescript
// request
interface {
  dischargeCode: str    // 排口编码
  monitorFactor: str    // 因子编码
}

// response
interface {

}
```

### GET|POST /page2/getModelPerformance 查询【分类预测】分类预警模型性能

```typescript
// request
interface {
  dischargeCode: str    // 排口编码
  monitorFactor: str    // 因子编码
}

// response
interface {

}
```

### GET|POST /page2/getExceedingPredictionResult 查询【分类预测】超标类别预测结果

```typescript
// request
interface {
  dischargeCode: str    // 排口编码
  monitorFactor: str    // 因子编码
}

// response
interface {

}
```

----

<p> by Armit <time> 2023/3/17 </time> </p>
