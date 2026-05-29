package com.pq.agent_service.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pq.agent_service.domain.po.User;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface UserMapper extends BaseMapper<User> {
}
